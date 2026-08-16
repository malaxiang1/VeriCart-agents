# Third-Party Notice

## Shopping environment

The project is built against:

- Repository: `https://github.com/YYHDBL/shopping-grpo-longhorizon.git`
- Pinned commit: `c3c178595eea835c18ba4515d553025014e52656`
- Role: ShopSimulator v2.1 environment, task format, evaluator, and veRL adapter base

The upstream checkout is not vendored in this release package. Clone it at the
pinned commit and apply the patch in `patches/shopping-grpo-c3c1785.patch`.
Check the upstream repository's current license and terms before redistribution.
At the time of the project audit, no license file was present in that checkout;
this package therefore does not grant rights to redistribute upstream code or
data.

## Model

Qwen3.5-2B weights are not included. Obtain them from the model provider under
their license and place them outside the repository. This project does not
redistribute model weights.

## Runtime dependencies

veRL, vLLM, PyTorch, Transformers, Ray, FlashInfer, PEFT, and the ShopSimulator
runtime retain their own licenses. Review and preserve their notices in any
deployment or redistribution.

## Project code

No open-source license is asserted by this package until the repository owner
chooses one. Select a license only after confirming that the upstream patch and
all included dependencies permit the intended distribution.
