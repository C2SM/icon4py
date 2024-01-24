# ICON-Liskov integration style-guide

The idea of the style-guide is to have a unique style, to make reading and maintaining as easy as possible.
The code should look as clean and concise as possible. Also it should be similar to the ACC style-guide: https://gitlab.dkrz.de/icon/wiki/-/wikis/GPU-development/ICON-OpenAcc-style-and-implementation-guide.

## General:

- indentation as original surrounding code.
- no empty line between following `!$DSL`-statements.

## Specific DSL statements:

- `!$DSL IMPORTS()` after last `USE` statement and before `IMPLICIT NONE`, one empty line before and after.
- `!$DSL DECLARE` after last variable declaration and before code block, one empty line before and after.
- `!$DSL START STENCIL` as close as possible to the start of the original stencil section, one empty line before, no empty line after.
- `!$DSL END STENCIL` as close as possible to the original stencil section, no empty line before, one empty line after.
- `!$DSL START FUSED STENCIL` as close as possible to the original stencil sections, one empty line before, one empty line after.
- `!$DSL END FUSED STENCIL` as close as possible to the end of the original stencil sections to be fused, one empty line before, one empty line after.
- `!$DSL INSERT` after `!$DSL START STENCIL`, no empty line before, one empty line after.
- `!$DSL START CREATE` after the `!$ACC CREATE` block, no empty line before and one empty line after.
