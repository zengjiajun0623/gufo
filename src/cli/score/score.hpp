#ifndef GUFO_CLI_SCORE_SCORE_HPP_
#define GUFO_CLI_SCORE_SCORE_HPP_

#include <cstddef>
#include <span>

namespace gufo::cli {

/// Teacher-forced corpus scoring: dumps per-position top-k logits, the
/// logsumexp, the argmax and the log-probability of the actual next token.
/// Used to gate quantization changes (top-1 agreement, truncated KL).
int RunScore(std::span<const char* const> args);
int RunDumpHidden(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_SCORE_SCORE_HPP_
