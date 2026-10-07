#include "src/cli/score/score.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace gufo::cli {
namespace {

namespace qfn = gufo::models::qwen38_flash_next;

struct ScoreOptions {
  std::string model_path;
  std::string corpus;
  std::string out{"score.json"};
  std::size_t top_k{64};
  std::size_t context{8192};
  std::size_t limit{0};
};

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::string{std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>()};
}

void EmitTopK(std::ofstream& out, std::span<const float> logits,
              std::size_t k, float lse) {
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
  out << "\"t\":[";
  for (std::size_t i = 0; i < k; ++i) {
    if (i != 0) out << ',';
    out << '[' << idx[i] << ',' << logits[idx[i]] - lse << ']';
  }
  out << ']';
}

int RunScoreImpl(const ScoreOptions& opt) {
  if (opt.model_path.empty()) throw std::runtime_error("--model is required");
  if (opt.corpus.empty()) throw std::runtime_error("--corpus is required");
  if (opt.context < 256) throw std::runtime_error("--context must be >= 256");
  if (opt.top_k == 0) throw std::runtime_error("--top-k must be >= 1");

  const auto corpus = json::parse(ReadFile(opt.corpus));
  if (!corpus.is_array()) throw std::runtime_error("corpus must be a JSON array of strings");
  std::vector<std::string> texts;
  for (const auto& v : corpus.items()) {
    if (v.is_string()) texts.push_back(v.str());
  }
  if (opt.limit != 0 && texts.size() > opt.limit) texts.resize(opt.limit);
  if (texts.empty()) throw std::runtime_error("corpus has no strings");

  std::string error;
  auto model = qfn::Model::Load(
      opt.model_path,
      qfn::ModelOptions{
          .max_context = static_cast<std::uint32_t>(opt.context),
          .mtp_model_path = "",
          .max_draft_tokens = 1,
      },
      &error);
  if (model == nullptr) {
    throw std::runtime_error("Model::Load failed: " + error);
  }
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      static_cast<std::uint32_t>(opt.context),
                                      &error);
  if (session == nullptr) {
    throw std::runtime_error("CreateSession failed: " + error);
  }

  std::ofstream out(opt.out, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + opt.out);
  out << "{\"top_k\":" << opt.top_k << ",\"texts\":[";

  double sum_logp = 0.0;
  std::size_t n_pos = 0;
  for (std::size_t t = 0; t < texts.size(); ++t) {
    auto ids = model->Tokenize(texts[t]);
    if (ids.size() > opt.context - 1) ids.resize(opt.context - 1);
    if (ids.size() < 2) {
      std::cerr << "skip text " << t << " (too short: " << ids.size()
                << " tokens)\n";
      continue;
    }
    if (!session->Sync(std::span<const std::int32_t>(ids).first(1), &error)) {
      throw std::runtime_error("Sync failed: " + error);
    }
    if (t != 0) out << ',';
    out << "{\"n\":" << ids.size() << ",\"pos\":[";
    for (std::size_t j = 0; j + 1 < ids.size(); ++j) {
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
      const std::int32_t next = ids[j + 1];
      const float lp_next =
          logits[static_cast<std::size_t>(next)] - lse;
      std::uint32_t argmax = 0;
      for (std::size_t v = 1; v < logits.size(); ++v) {
        if (logits[v] > logits[argmax]) argmax = static_cast<std::uint32_t>(v);
      }
      if (j != 0) out << ',';
      out << "{\"a\":" << argmax << ",\"nx\":" << next
          << ",\"lse\":" << lse << ",\"lp\":" << lp_next << ',';
      EmitTopK(out, logits, opt.top_k, lse);
      out << '}';
      sum_logp += lp_next;
      ++n_pos;
    }
    out << "]}";
    if ((t + 1) % 5 == 0) {
      std::cerr << "scored " << t + 1 << '/' << texts.size() << " texts, "
                << n_pos << " positions\n";
    }
  }
  out << "]}";
  out.close();
  std::cout << "scored " << n_pos << " positions over " << texts.size()
            << " texts; mean next-token logp = "
            << (n_pos ? sum_logp / static_cast<double>(n_pos) : 0.0) << '\n';
  return 0;
}

}  // namespace

int RunScore(std::span<const char* const> args) {
  ScoreOptions opt;
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
    } else if (a == "--corpus") {
      opt.corpus = std::string(need(a));
    } else if (a == "--out") {
      opt.out = std::string(need(a));
    } else if (a == "--top-k") {
      opt.top_k = std::stoul(std::string(need(a)));
    } else if (a == "--context") {
      opt.context = std::stoul(std::string(need(a)));
    } else if (a == "--limit") {
      opt.limit = std::stoul(std::string(need(a)));
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: gufo score --model M.gguf --corpus texts.json "
                   "[--out score.json] [--top-k 64] [--context 8192] "
                   "[--limit N]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(a));
    }
  }
  try {
    return RunScoreImpl(opt);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::cli
