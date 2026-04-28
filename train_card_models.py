#!/usr/bin/env python3
"""Offline trainer for rank/suit PyTorch symbol models."""

from card_symbol_model import train_rank_and_suit_models


def main() -> None:
    def progress(msg: str) -> None:
        print(msg, flush=True)

    results = train_rank_and_suit_models(
        include_user_data=True,
        bootstrap_if_missing=True,
        epochs=10,
        lr=1e-3,
        batch_size=64,
        progress=progress,
    )

    rank = results["rank"]
    suit = results["suit"]
    print(
        f"Training complete. rank_samples={rank.samples} rank_loss={rank.final_loss:.4f} "
        f"suit_samples={suit.samples} suit_loss={suit.final_loss:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
