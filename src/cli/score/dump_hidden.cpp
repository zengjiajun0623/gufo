// `gufo dump-hidden`: teacher-forced training-data dump for MTP draft-head
// self-distillation. For every corpus position t (token stream x, position j
// = 0..len-2):
//   hidden row h[j]   — the trunk hyper-connection residual the MTP block
//                       consumes as `hidden` (HcDim floats, written bf16)
//   trunk next dist   — top-k (id, logp) + lse (same as `score`)
//   draft argmax      — the GPU MTP head's greedy token for (h[j], x[j+1])
//   target            — x[j+1]
// With --reference, additionally runs the float32 CPU ReferenceModel over the
// same texts and writes ref_hidden.f32 + ref_mtp_argmax.i32 for acceptance.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/score/score.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::cli {
namespace {

namespace q = gufo::models::qwen38_flash_next;

struct DumpOptions {
  std::string model_path;
  std::string mtp_path;
  std::string corpus;
  std::string out_dir{"dump"};
  std::size_t context{8192};
  std::size_t top_k{64};
  std::size_t limit{0};
  bool reference{false};
  std::size_t batch{0};  // >0: prefill rows in chunks (GUFO_DUMP_ROWS); no draft argmax
};

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::string{std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>()};
}

std::uint16_t ToBf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  std::uint16_t b = static_cast<std::uint16_t>(u >> 16);
  const std::uint32_t frac = u & 0xFFFFu;
  if (frac > 0x8000u || (frac == 0x8000u && (b & 1u) != 0u)) {
    ++b;  // round to nearest even on the truncated half
  }
  return b;
}

void EmitTopK(std::ofstream& ids_out, std::ofstream& val_out,
              std::span<const float> logits, std::size_t k, float lse) {
  const std::size_t vocab = logits.size();
  std::vector<std::uint32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0u);
  k = std::min(k, vocab);
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
                    idx.end(), [&](std::uint32_t a, std::uint32_t b) {
                      return logits[a] > logits[b];
                    });
  std::sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
            [&](std::uint32_t a, std::uint32_t b) {
              return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
            });
  for (std::size_t i = 0; i < k; ++i) {
    const std::int32_t id = static_cast<std::int32_t>(idx[i]);
    const float lp = logits[idx[i]] - lse;
    ids_out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    val_out.write(reinterpret_cast<const char*>(&lp), sizeof(lp));
  }
}

struct RowStats {
  float lse{0};
  std::vector<std::int32_t> ids;
  std::vector<float> lp;
};

RowStats TopKRow(std::span<const float> logits, std::size_t k) {
  RowStats r;
  float max_v = logits[0];
  for (float v : logits) max_v = std::max(max_v, v);
  double sum = 0.0;
  for (float v : logits) sum += std::exp(static_cast<double>(v - max_v));
  r.lse = max_v + static_cast<float>(std::log(sum));
  const std::size_t vocab = logits.size();
  std::vector<std::uint32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0u);
  k = std::min(k, vocab);
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
                    idx.end(), [&](std::uint32_t a, std::uint32_t b) {
                      return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
                    });
  for (std::size_t i = 0; i < k; ++i) {
    r.ids.push_back(static_cast<std::int32_t>(idx[i]));
    r.lp.push_back(logits[idx[i]] - r.lse);
  }
  return r;
}

int RunDumpHiddenImpl(const DumpOptions& opt) {
  if (opt.model_path.empty()) throw std::runtime_error("--model is required");
  if (opt.mtp_path.empty()) throw std::runtime_error("--mtp-model is required");
  if (opt.corpus.empty()) throw std::runtime_error("--corpus is required");

  const auto corpus = json::parse(ReadFile(opt.corpus));
  if (!corpus.is_array()) {
    throw std::runtime_error("corpus must be a JSON array of strings");
  }
  std::vector<std::string> texts;
  for (const auto& v : corpus.items()) {
    if (v.is_string()) texts.push_back(v.str());
  }
  if (opt.limit != 0 && texts.size() > opt.limit) texts.resize(opt.limit);
  if (texts.empty()) throw std::runtime_error("corpus has no strings");

  std::string error;
  if (opt.batch != 0) {
    if (opt.reference) throw std::runtime_error("--batch excludes --reference");
    ::setenv("GUFO_DUMP_ROWS", std::to_string(opt.batch).c_str(), 1);
  }
  auto model = q::Model::Load(
      opt.model_path,
      q::ModelOptions{
          .max_context = static_cast<std::uint32_t>(opt.context),
          .mtp_model_path = opt.mtp_path,
          .max_draft_tokens = 7,
      },
      &error);
  if (model == nullptr) {
    throw std::runtime_error("Model::Load failed: " + error);
  }
  auto session = model->CreateSession(
      gufo::core::SessionMode::kSpeculative,
      static_cast<std::uint32_t>(opt.context), &error);
  if (session == nullptr) {
    throw std::runtime_error("CreateSession failed: " + error);
  }

  // Optional float32 CPU reference (slow; only for short acceptance texts).
  std::unique_ptr<gufo::core::GgufReader> ref_reader;
  std::optional<q::ModelWeights> ref_weights;
  std::unique_ptr<q::NgramTable> ref_ngram;
  q::MtpWeights ref_mtp;
  // ref_mtp points into this mapping: keep the reader open while it is used.
  std::unique_ptr<gufo::core::GgufReader> mtp_reader;
  bool have_ref = false;
  if (opt.reference) {
    ref_reader = gufo::core::GgufReader::OpenFile(opt.model_path, &error);
    if (!ref_reader) throw std::runtime_error("reference open failed: " + error);
    ref_weights = q::ModelWeights::Bind(*ref_reader, &error);
    if (!ref_weights) throw std::runtime_error("reference bind failed: " + error);
    const auto& c = ref_weights->config;
    if (c.ple_layer >= 0) {
      const auto& t = ref_weights->ple_table;
      ref_ngram = q::NgramTable::Open(
          ref_reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
          t.rows, c.ple_head_dim, t.type, &error);
      if (!ref_ngram) throw std::runtime_error("n-gram table failed: " + error);
    }
    mtp_reader = gufo::core::GgufReader::OpenFile(opt.mtp_path, &error);
    if (!mtp_reader) throw std::runtime_error("mtp open failed: " + error);
    auto mtp = q::MtpWeights::Bind(*mtp_reader, ref_weights->config, &error);
    if (!mtp) throw std::runtime_error("mtp bind failed: " + error);
    ref_mtp = std::move(*mtp);
    have_ref = true;
  }

  std::filesystem::create_directories(opt.out_dir);
  const auto path = [&](const char* name) {
    return opt.out_dir + "/" + name;
  };
  std::ofstream tokens_out(path("tokens.i32"), std::ios::binary);
  std::ofstream hidden_out(path("hidden.bf16"), std::ios::binary);
  std::ofstream topk_ids(path("topk.bin"), std::ios::binary);
  std::ofstream topk_val(path("topk_val.f32"), std::ios::binary);
  std::ofstream lse_out(path("lse.f32"), std::ios::binary);
  std::ofstream draft_out(path("draft_argmax.i32"), std::ios::binary);
  std::ofstream ref_hidden;
  std::ofstream ref_argmax;
  std::ofstream ref_mtp_logits;  // full vocab per position: torch parity gate
  if (have_ref) {
    ref_hidden.open(path("ref_hidden.f32"), std::ios::binary);
    ref_argmax.open(path("ref_mtp_argmax.i32"), std::ios::binary);
    ref_mtp_logits.open(path("ref_mtp_logits.f32"), std::ios::binary);
  }
  if (!tokens_out || !hidden_out || !topk_ids || !topk_val || !lse_out ||
      !draft_out) {
    throw std::runtime_error("cannot write outputs in " + opt.out_dir);
  }

  json::Value offsets = json::Value::array();
  std::size_t positions = 0;
  std::size_t token_count = 0;
  const auto t0 = std::chrono::steady_clock::now();

  for (std::size_t t = 0; t < texts.size(); ++t) {
    auto ids = model->Tokenize(texts[t]);
    if (ids.size() > opt.context - 1) ids.resize(opt.context - 1);
    if (ids.size() < 2) {
      std::cerr << "skip text " << t << " (too short)\n";
      continue;
    }
    if (!session->Sync(std::span<const std::int32_t>(ids).first(1), &error)) {
      throw std::runtime_error("Sync failed: " + error);
    }
    std::unique_ptr<q::ReferenceModel> ref;
    std::vector<float> hc;
    std::vector<float> ref_logits;
    if (have_ref) {
      ref = std::make_unique<q::ReferenceModel>(
          *ref_weights, ref_ngram.get(),
          static_cast<std::uint32_t>(ids.size()) + 1);
      hc.resize(ref_weights->config.HcDim());
      ref_logits.resize(ref_weights->config.vocab_size);
    }
    json::Value entry = json::Value::object();
    entry["start"] = static_cast<double>(token_count);
    entry["len"] = static_cast<double>(ids.size());
    offsets.push_back(entry);
    if (opt.batch != 0) {
      // Row 0 comes from Sync; rows 1..n-2 from chunked prefill of ids[1..n-2],
      // emitted per segment so host memory stays ~kSeg rows of full logits.
      constexpr std::size_t kSeg = 256;
      const std::size_t vocab = model->VocabSize();
      const std::size_t hcd = model->config().HcDim();
      const std::size_t rows = ids.size() - 1;
      const std::int32_t none = -1;
      const std::size_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
      std::vector<float> lg, hd;
      for (std::size_t r0 = 0; r0 < rows;) {
        std::size_t n;
        if (r0 == 0) {
          n = 1;
          const auto l0 = session->Logits();
          lg.assign(l0.begin(), l0.end());
          hd = session->CopyLastHidden();
          if (hd.size() != hcd) throw std::runtime_error("CopyLastHidden failed");
        } else {
          n = std::min(kSeg, rows - r0);
          if (!session->DumpForward(std::span<const std::int32_t>(ids).subspan(r0, n),
                                    &lg, &hd, &error))
            throw std::runtime_error("DumpForward failed: " + error);
        }
        std::vector<RowStats> st(n);
        std::vector<std::thread> pool;
        for (std::size_t w = 0; w < nt; ++w)
          pool.emplace_back([&, w] {
            for (std::size_t j = w; j < n; j += nt)
              st[j] = TopKRow(std::span<const float>(lg).subspan(j * vocab, vocab), opt.top_k);
          });
        for (auto& th : pool) th.join();
        for (std::size_t j = 0; j < n; ++j) {
          tokens_out.write(reinterpret_cast<const char*>(&ids[r0 + j + 1]), sizeof(std::int32_t));
          for (std::size_t i = 0; i < hcd; ++i) {
            const std::uint16_t b = ToBf16(hd[j * hcd + i]);
            hidden_out.write(reinterpret_cast<const char*>(&b), sizeof(b));
          }
          topk_ids.write(reinterpret_cast<const char*>(st[j].ids.data()),
                         static_cast<std::streamsize>(st[j].ids.size() * sizeof(std::int32_t)));
          topk_val.write(reinterpret_cast<const char*>(st[j].lp.data()),
                         static_cast<std::streamsize>(st[j].lp.size() * sizeof(float)));
          lse_out.write(reinterpret_cast<const char*>(&st[j].lse), sizeof(float));
          draft_out.write(reinterpret_cast<const char*>(&none), sizeof(none));
        }
        positions += n;
        r0 += n;
      }
    }
    for (std::size_t j = 0; opt.batch == 0 && j + 1 < ids.size(); ++j) {
      if (j > 0) {
        if (!session->Evaluate(ids[j], &error)) {
          throw std::runtime_error("Evaluate failed: " + error);
        }
      }
      const auto logits = session->Logits();
      if (logits.empty()) throw std::runtime_error("no logits at position");
      float max_v = logits[0];
      for (std::size_t v = 1; v < logits.size(); ++v) {
        max_v = std::max(max_v, logits[v]);
      }
      double sum = 0.0;
      for (float v : logits) sum += std::exp(static_cast<double>(v - max_v));
      const float lse = max_v + static_cast<float>(std::log(sum));

      const auto h = session->CopyLastHidden();
      if (h.empty()) throw std::runtime_error("CopyLastHidden failed");
      std::int32_t gpu_argmax = -1;
      if (!session->DraftProbe(ids[j + 1], &gpu_argmax, &error)) {
        throw std::runtime_error("DraftProbe failed: " + error);
      }

      tokens_out.write(reinterpret_cast<const char*>(&ids[j + 1]),
                       sizeof(ids[j + 1]));
      for (float v : h) {
        const std::uint16_t b = ToBf16(v);
        hidden_out.write(reinterpret_cast<const char*>(&b), sizeof(b));
      }
      EmitTopK(topk_ids, topk_val, logits, opt.top_k, lse);
      lse_out.write(reinterpret_cast<const char*>(&lse), sizeof(lse));
      draft_out.write(reinterpret_cast<const char*>(&gpu_argmax),
                      sizeof(gpu_argmax));
      if (have_ref) {
        if (!ref->Step(ids[j], {}, hc, &error)) {
          throw std::runtime_error("reference Step failed: " + error);
        }
        ref_hidden.write(reinterpret_cast<const char*>(hc.data()),
                         static_cast<std::streamsize>(hc.size() *
                                                      sizeof(float)));
        if (!ref->MtpStep(ref_mtp, ids[j + 1], hc, ref_logits, nullptr, nullptr,
                          &error)) {
          throw std::runtime_error("reference MtpStep failed: " + error);
        }
        const std::int32_t ref_arg =
            static_cast<std::int32_t>(std::max_element(ref_logits.begin(),
                                                       ref_logits.end()) -
                                      ref_logits.begin());
        ref_argmax.write(reinterpret_cast<const char*>(&ref_arg),
                         sizeof(ref_arg));
        ref_mtp_logits.write(
            reinterpret_cast<const char*>(ref_logits.data()),
            static_cast<std::streamsize>(ref_logits.size() * sizeof(float)));
      }
      ++positions;
    }
    token_count += ids.size();
    if ((t + 1) % 5 == 0 || t + 1 == texts.size()) {
      std::cerr << "dumped " << t + 1 << '/' << texts.size() << " texts, "
                << positions << " positions\n";
    }
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  std::ofstream offsets_out(path("offsets.json"), std::ios::binary);
  offsets_out << offsets.dump() << "\n";
  std::ofstream thr(path("throughput.json"), std::ios::binary);
  thr << "{\"positions\":" << positions << ",\"seconds\":" << seconds << "}\n";
  std::cout << "dumped " << positions << " positions (" << token_count
            << " tokens) in " << seconds << " s -> " << opt.out_dir << '\n';
  return 0;
}

}  // namespace

int RunDumpHidden(std::span<const char* const> args) {
  DumpOptions opt;
  try {
    for (std::size_t i = 0; i < args.size(); ++i) {
      const std::string_view a = args[i];
      auto need = [&](std::string_view name) -> std::string_view {
        if (i + 1 >= args.size()) {
          throw std::runtime_error(std::string(name) + " needs a value");
        }
        return args[++i];
      };
      if (a == "--model") {
        opt.model_path = std::string(need(a));
      } else if (a == "--mtp-model") {
        opt.mtp_path = std::string(need(a));
      } else if (a == "--corpus") {
        opt.corpus = std::string(need(a));
      } else if (a == "--out") {
        opt.out_dir = std::string(need(a));
      } else if (a == "--context") {
        opt.context = std::stoul(std::string(need(a)));
      } else if (a == "--top-k") {
        opt.top_k = std::stoul(std::string(need(a)));
      } else if (a == "--limit") {
        opt.limit = std::stoul(std::string(need(a)));
      } else if (a == "--batch") {
        opt.batch = std::stoul(std::string(need(a)));
      } else if (a == "--reference") {
        opt.reference = true;
      } else if (a == "--help" || a == "-h") {
        std::cout
            << "Usage: gufo dump-hidden --model M.gguf --mtp-model T.gguf "
               "--corpus texts.json --out DIR [--context 8192] [--top-k 64] "
               "[--limit N] [--reference] [--batch ROWS]\n";
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + std::string(a));
      }
    }
    return RunDumpHiddenImpl(opt);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::cli
