// SnapKV R1 原型：观测非 FA attention 链的 kq softmax 分布
// （GGML_OP_SOFT_MAX 节点仅由非 FA 链产生——FA 是融合 op，无独立 softmax）
// 用途：验证"重要性长尾"前提（SnapKV 类 KV 压缩的可行性判据）
// 用法示例：llama-attn-stats -m model.gguf -f prompt.txt -c 4096 -b 4096 -n 64 -t 4

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct attn_stats {
    std::vector<double> acc;   // [n_kv] 全局累计（跨层/步/head）
    std::vector<std::vector<double>> step_dist;   // [step] -> [pos]，该步分布（层/head 合并）
    int64_t n_captured = 0;
    int64_t n_kv_max   = 0;
};

static bool attn_stats_cb(struct ggml_tensor * t, bool ask, void * ud_) {
    auto * ud = (attn_stats *) ud_;

    if (ask) {
        return t->op == GGML_OP_SOFT_MAX;
    }

    const int64_t n_kv = t->ne[0];
    const int64_t n_q  = t->ne[1];

    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    if ((int64_t) ud->acc.size() < n_kv) {
        ud->acc.resize(n_kv, 0.0);
    }
    if (ud->step_dist.empty()) {
        ud->step_dist.push_back({});
    }
    auto & sd = ud->step_dist.back();
    if ((int64_t) sd.size() < n_kv) {
        sd.resize(n_kv, 0.0);
    }

    const float * p = buf.data();
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t q = 0; q < n_q; q++) {
                for (int64_t i = 0; i < n_kv; i++) {
                    ud->acc[i] += p[i];
                    sd[i]      += p[i];
                }
                p += n_kv;
            }
        }
    }

    ud->n_captured++;
    ud->n_kv_max = std::max(ud->n_kv_max, n_kv);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    attn_stats ud;
    params.cb_eval = attn_stats_cb;
    params.cb_eval_user_data = &ud;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return 1;
    }
    LOG_INF("%s: prompt tokens = %zu\n", __func__, tokens.size());

    // prefill（大 M 走 FA，无观测）
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : prefill failed\n", __func__);
        return 1;
    }

    // 逐 token 解码（M=1，深 KV → 非 FA 链被观测）
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    const int n_gen = params.n_predict > 0 ? params.n_predict : 64;
    std::string gen;
    for (int i = 0; i < n_gen; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        gen += common_token_to_piece(ctx, id);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        ud.step_dist.push_back({});
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            LOG_ERR("%s : decode failed at step %d\n", __func__, i);
            break;
        }
    }
    common_sampler_free(smpl);

    LOG_INF("%s: gen: %s\n", __func__, gen.c_str());

    // 统计输出
    LOG_INF("\n=== attn stats ===\n");
    LOG_INF("captured softmax evals = %lld, n_kv_max = %lld\n",
            (long long) ud.n_captured, (long long) ud.n_kv_max);

    if (ud.n_captured > 0 && !ud.acc.empty()) {
        std::vector<double> v = ud.acc;
        double total = 0.0;
        for (double x : v) total += x;
        std::sort(v.begin(), v.end(), std::greater<double>());
        const int64_t n = (int64_t) v.size();
        LOG_INF("positions = %lld, total mass = %.3f\n", (long long) n, total);
        for (double pct : {1.0, 5.0, 10.0, 25.0, 50.0}) {
            int64_t k = std::max<int64_t>(1, (int64_t) (n * pct / 100.0));
            double s = 0.0;
            for (int64_t i = 0; i < k; i++) s += v[i];
            LOG_INF("top %5.1f%% (%6lld pos): mass share = %.4f\n",
                    pct, (long long) k, total > 0 ? s / total : 0.0);
        }
    }

    // R2：因果版——观察窗（前 s 步）选出的 top-25% 位置，在未来（s..end）同域内的质量保留
    {
        const int64_t n_steps = (int64_t) ud.step_dist.size();
        LOG_INF("\n=== window prediction (causal) ===\n");
        for (int64_t s : { (int64_t) 16, (int64_t) 32, (int64_t) 48 }) {
            if (s >= n_steps) {
                continue;
            }
            std::vector<double> obs, fut;
            for (int64_t t = 0; t < s; t++) {
                const auto & d = ud.step_dist[t];
                if ((int64_t) obs.size() < (int64_t) d.size()) obs.resize(d.size(), 0.0);
                for (size_t i = 0; i < d.size(); i++) obs[i] += d[i];
            }
            for (int64_t t = s; t < n_steps; t++) {
                const auto & d = ud.step_dist[t];
                if ((int64_t) fut.size() < (int64_t) d.size()) fut.resize(d.size(), 0.0);
                for (size_t i = 0; i < d.size(); i++) fut[i] += d[i];
            }
            const int64_t n = (int64_t) obs.size();
            const int64_t k = std::max<int64_t>(1, (int64_t) (n * 0.25));
            std::vector<int64_t> idx(n);
            for (int64_t i = 0; i < n; i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return obs[a] > obs[b]; });
            std::vector<char> sel(n, 0);
            for (int64_t i = 0; i < k; i++) sel[idx[i]] = 1;

            const int64_t lim = std::min<int64_t>(n, (int64_t) fut.size());
            double m_sel = 0.0, m_tot = 0.0;
            for (int64_t i = 0; i < lim; i++) {
                if (sel[i]) m_sel += fut[i];
                m_tot += fut[i];
            }
            // oracle：未来同域内的 top-k 占比
            std::vector<double> fv(fut.begin(), fut.begin() + lim);
            std::sort(fv.begin(), fv.end(), std::greater<double>());
            double fo = 0.0;
            for (int64_t i = 0; i < std::min<int64_t>(k, lim); i++) fo += fv[i];

            LOG_INF("obs=%lld steps, n=%lld: selected-25%% future-mass=%.4f | oracle-top25%%=%.4f\n",
                    (long long) s, (long long) n,
                    m_tot > 0 ? m_sel / m_tot : 0.0,
                    m_tot > 0 ? fo / m_tot : 0.0);
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
