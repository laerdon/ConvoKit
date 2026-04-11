"""
end-to-end training script for a TransformerDecoderModel forecaster with a DeferralDecisionPolicy.

flow:
  1. load the CGA-CMV corpus
  2. build a DeferralDecisionPolicy backed by an UnslothUtteranceSimulatorModel
  3. build a TransformerDecoderModel (forecaster backbone) with that policy attached
  4. wrap both in a Forecaster
  5. fit: LoRA fine-tune the forecaster, then fit the decision policy on the val set
  6. evaluate on the test set and print metrics

usage:
  python train_deferral.py [--device cuda] [--gpu 0]
"""

import argparse
import os
import sys

# ensure the repo root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main(args):
    import convokit
    from convokit import Corpus, Forecaster, download
    from convokit.forecaster.TransformerDecoderModel import TransformerDecoderModel
    from convokit.forecaster.TransformerForecasterConfig import TransformerForecasterConfig
    from convokit.decisionpolicy import DeferralDecisionPolicy, ThresholdDecisionPolicy

    # ------------------------------------------------------------------ #
    # 1. corpus
    # ------------------------------------------------------------------ #
    print("[info] loading corpus...")
    corpus = Corpus(
        filename=download(
            "conversations-gone-awry-cmv-corpus",
            data_dir=args.data_dir,
        )
    )

    labeler = "has_removed_comment"

    # ------------------------------------------------------------------ #
    # 2. context selectors
    # ------------------------------------------------------------------ #
    def train_selector(ctx):
        """last context of every train conversation (matches original craft/llm training setup)"""
        convo = ctx.current_utterance.get_conversation()
        return (
            convo.meta.get("split") == "train"
            and len(ctx.future_context) == 0
        )

    def val_selector(ctx):
        return ctx.current_utterance.get_conversation().meta.get("split") == "val"

    def test_selector(ctx):
        convo = ctx.current_utterance.get_conversation()
        convo_len = len(convo.get_chronological_utterance_list())
        return (
            convo.meta.get("split") == "test"
            # exclude the very last context (the toxic turn itself)
            and len(ctx.context) < convo_len
        )

    
    # 3. simulator model
    #  #
    # 4. decision policy
    policy = ThresholdDecisionPolicy(
        threshold=0.5926666259765625,
    )

    # 5. forecaster model
    print("[info] loading forecaster model...")
    config = TransformerForecasterConfig(
        output_dir=args.output_dir,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        random_seed=args.seed,
        context_mode="normal",
        device=args.device,
    )

    forecaster_model = TransformerDecoderModel(
        model_name_or_path=args.forecaster_model,
        config=config,
        decision_policy=policy,
    )

    # 6. forecaster wrapper
    forecaster = Forecaster(
        forecaster_model=forecaster_model,
        labeler=labeler,
    )

    # 7. fit
    # print("[info] fitting forecaster (belief estimator + decision policy)...")
    # forecaster.fit(
    #     corpus=corpus,
    #     context_selector=train_selector,
    #     val_context_selector=val_selector,
    # )

    forecaster.fit_decision_policy(
        corpus=corpus,
        context_selector=train_selector,
        val_context_selector=val_selector,
    )

    # print(forecaster.forecaster_model.decision_policy.threshold)

    # # 8. evaluate on test set
    # print("[info] running transform on test set...")
    # corpus = forecaster.transform(
    #     corpus=corpus,
    #     context_selector=test_selector,
    # )

    # print("[info] computing metrics...")
    # forecaster.summarize(
    #     corpus=corpus,
    #     selector=lambda convo: convo.meta.get("split") == "test",
    # )

    # optional: inspect a few utterances with stored simulations
    if args.store_simulations:
        print("\n[info] sample utterances with stored simulations:")
        shown = 0
        for utt in corpus.iter_utterances():
        # show only utterances that were forecasted and have sim_replies
            if (
                utt.meta.get("forecast") is not None
                and utt.meta.get("sim_replies") is not None
            ):
                print("---")
                print("text          :", utt.text[:120])
                print("forecast_prob :", utt.meta["forecast_prob"])
                print("forecast      :", utt.meta["forecast"])
                print("sim_replies   :", utt.meta["sim_replies"][:2])
                print("sim_probs     :", utt.meta["sim_replies_forecast_probs"][:2])
                shown += 1
                if shown >= 3:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train forecaster with DeferralDecisionPolicy")

    # paths
    parser.add_argument("--forecaster-model", required=True,
                        help="hf model name or local path for the decoder forecaster")
    parser.add_argument("--simulator-model", required=True,
                        help="hf model name or local path for the utterance simulator")
    parser.add_argument("--output-dir", default="./deferral_output",
                        help="directory to save checkpoints and predictions")
    parser.add_argument("--data-dir", default="./",
                        help="directory to download/find the corpus")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)

    # deferral policy hyperparams
    parser.add_argument("--num-simulations", type=int, default=10,
                        help="number of simulated branches per context")
    parser.add_argument("--tau", type=int, default=5,
                        help="minimum simulated branches above threshold to intervene")

    # misc
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=3,
                        help="which gpu to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--store-simulations", action="store_true",
                        help="write simulated replies and their forecast probs to corpus metadata")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    main(args)
