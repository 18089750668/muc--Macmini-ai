"""Per-round accounting. The one number that matters is useful_inflight:
throughput over WAN is useful in-flight work per round-trip, not pipe fullness.
"""
from dataclasses import dataclass, field


@dataclass
class Metrics:
    rounds: list = field(default_factory=list)

    def round(self, tau_h, tau_t, latency, n_in, accepted):
        self.rounds.append(dict(tau_h=tau_h, tau_t=tau_t, latency=latency,
                                n_in=n_in, accepted=accepted))

    def summary(self, wall_s, tokens):
        decode = [r for r in self.rounds[1:]] or self.rounds
        taus = [(r["tau_h"] + r["tau_t"]) for r in decode]
        tavg = sum(taus) / len(taus) if taus else 0.0
        lats = [r["latency"] for r in decode]
        lavg = sum(lats) / len(lats) if lats else 0.0
        acc = sum(r["accepted"] for r in decode)
        g = acc / len(decode) if decode else 0.0
        frames_per_token = len(decode) / max(tokens, 1)
        useful = tavg / frames_per_token if frames_per_token else 0.0
        tau_h = sum(r["tau_h"] for r in decode) / len(decode) if decode else 0.0
        tau_t = sum(r["tau_t"] for r in decode) / len(decode) if decode else 0.0
        return dict(
            rounds=len(self.rounds), tokens=tokens, wall_s=wall_s,
            prefill_ms=self.prefill_ms,
            tok_s=tokens / wall_s if wall_s else 0.0,
            tavg_ms=tavg, latency_ms=lavg, g=g,
            tau_h_ms=tau_h, tau_t_ms=tau_t,
            hop_ms=max(lavg - tau_t, 0.0),
            frames_per_token=frames_per_token, useful_inflight=useful,
            max_inflight=max(r["n_in"] for r in decode) if decode else 0,
        )
