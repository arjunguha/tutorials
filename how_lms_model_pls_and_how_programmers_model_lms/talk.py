import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="full",
    app_title="Towards Understanding How Language Models Model Programming Languages and How Programmers Model Language Models",
    layout_file="layouts/talk.slides.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import itertools
    import pandas as pd
    import tempfile
    from typing import List, Optional
    from tqdm.auto import tqdm
    import torch
    import litellm
    from nnsight import LanguageModel
    from pathlib import Path
    from more_nnsight.standard_model import StandardModel
    from more_nnsight.residuals import last_token_residuals, pca_plot
    from abstractions.iterables import batches
    from abstractions.storage import disk_cache
    from abstractions.async_abstractions import run_bounded
    import bounded_subprocess.bounded_subprocess_async as bounded_subprocess_async
    return (
        LanguageModel,
        List,
        Optional,
        Path,
        StandardModel,
        batches,
        bounded_subprocess_async,
        disk_cache,
        itertools,
        last_token_residuals,
        litellm,
        mo,
        pca_plot,
        pd,
        run_bounded,
        tempfile,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How Language Models Model Programming Languages and <br>How Programmers Model Language Models

    Arjun Guha, *Northeastern University*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prelude: Benchmarking Programming Tasks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating Code LLMs

    *Some results from 2021--2022*:

    > "We introduce Codex ... on HumanEval, a new evaluation set we release ... our model solves 28.8% of the problems"

    Chen et al. Evaluating Large Language Models Trained on Code, 2021

    > "can synthesize solutions to 59.6% of the problems from [Mostly Basic Python Programs] MBPP"

      Austin et al. Program Synthesis with Large Language Models, 2021

    > "CodeGen outperforms OpenAI’s Codex on the HumanEval benchmark"

    Nijkamp et al. A Conversational Paradigm for Program Synthesis, 2022

    > "We report the performance of the 6.7B model on the HumanEval benchmark"

    Fried et al. InCoder: A Generative Model for Code Infilling and Synthesis, 2022
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An Example Problem from HumanEval

    **Prompt:**

    ```python
    def vowels_count(s):
        '''
        Write a function vowels_count which takes a string representing a word as input
        and returns the number of vowels in the string.  Vowels in this case are 'a', 'e',
        'i', 'o', 'u'. Here, 'y' is also a vowel, but only when it is at the end of the
        given word.
        '''
    ```

    **Hidden unit tests:**

    ```python
    assert vowels_count("hello") == 2
    assert vowels_count("lazy") == 2
    assert vowels_count("greedy") == 3
    ...
    ```

    **Summary of benchmarks**:

     Model         | Evaluation
    :--------------|:----------
    Chen et al.    | 164 Python prompts
    Austin et al.  | 400+ Python prompts
    Nijkamp et al. | 50 Python prompts
    Fried et al.   | more prompts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Our Approach: Rule-Based Translation of Prompt and Tests to 18+ Other PLs

    **Prompt translation and type inference:**

    ```python
    from typing import List, Tuple, Optional

    def lsi(lst):
        '''
        Create a function that returns a tuple (a, b), where 'a' is
        the largest of negative integers, and 'b' is the smallest
        of positive integers in a list.
        If there is no negative or positive integers, return them as None.
        '''
    ```
    ➡️
    ```rust
    /// Create a function that returns a tuple (a, b), where 'a' is
    /// the largest of negative integers, and 'b' is the smallest
    /// of positive integers in a vector.
    /// If there is no negative or positive integers, return them as None.
    fn lsi(lst: Vec<isize>) -> (Option<isize>, Option<isize>) {
    ```

    **Test translation**

    ```python
    assert lsi([2,4,1,3,5,7]) == (None, 1)
    ```
    ➡️
    ```rust
    assert_eq!(lsi(vec![2,4,1,3,5,7]), (None, Some(1)))
    ```

    1. The "type inference" is from the concrete Python values in the Python test suite.
    2. We inject *T*-typed values into `Optional<T>` as needed.

    *From:* Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee,
    Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. [MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation](https://ieeexplore.ieee.org/document/10103177). *IEEE Transactions on Software Engineering (TSE)*, 2023
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.output.append(mo.md("## Evaluation with MultiPL-E (mid-2023)"))
    mo.output.append(mo.image("public/multipl_e_original_results.png", width="50%"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation with MultiPL-E (mid-2025)

    | Model                  | MultiPL-E (Pass@1) |
    |-------------------------|--------------------|
    | Kimi-K2-Instruct        | 0.86               |
    | DeepSeek-V3-0324        | 0.83               |
    | Qwen3-235B-A22B         | 0.78               |
    | Claude Sonnet 4         | 0.89               |
    | Claude Opus 4           | **0.90**           |
    | GPT-4.1                 | 0.87               |
    | Gemini 2.5 Flash        | 0.86               |

    *Source:* Kimi Team. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534), 2025:

    ---

    **10% of HumanEval problems are faulty (bad tests, ambiguous instructions, etc.)**

    *Source*: Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. NeurIPS, 2023
    """)
    return


@app.cell
def _(pd):
    agnostic_test_set = pd.read_json("humaneval_agnostic.jsonl", lines=True).to_dict(orient="records")
    return (agnostic_test_set,)


@app.cell(hide_code=True)
def _(agnostic_test_set, mo):
    mo.md(rf"""
    ## The Agnostics Approach

    1. You give newer models a harder task: write a complete program to do a task, with detailed instructions
       on how to format input and output. *The following is a HumanEval prompt reformulated.*

    > {agnostic_test_set[0]["task"].replace("\n", "\n>")}


    2. Add *"solve this problem in OCaml"* to the prompt above. Most of the test harness can be language-neutral.
       (We only need to know how to build and run code in the target language.)

    3. In many cases, you can have a capable LLM transform language-specific tasks and tests into language-agnostic tasks and tests

    Applied to a benchmark, you get a a language-agnostic benchmark. Applied to training items, you get language-agnostic tasks for reinforcement learning.

    *From:* Aleksander Boruch-Gruszecki, Yangtian Zi, Zixuan Wu, Tejas Oberoi, Carolyn Jane Anderson, Joydeep Biswas, Arjun Guha. [Agnostics: Learning to Code in Any Programming Language via Reinforcement with a Universal Learning Environment](https://arxiv.org/abs/2508.04865), 2025
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agnostics Results on OCaml and Fortran

    - Original train set: Guilherme et al. [Open-R1 CodeForces](https://huggingface.co/datasets/open-r1/codeforces). 2025
    - Original benchmark: Jain et al. [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974). *ICLR*, 2025

    Evaluation with reformulatd problems:

    | Model                       |         OCaml |        Fortran |
    |-----------------------------|---------------|----------------|
    | Sonnet 4                    | 0.06          | 0.06           |
    | Llama 3.3 70B Instruct      | 0.07          | 0.03           |
    | Qwen 3 32B                  | 0.02          | 0.01           |
    | Qwen 3 4B                   | 0.01          | 0.00           |
    | Qwen 3 4B Agnostics         | 0.07          | 0.15           |

    - We train separate Agnostics models for each language
    - The paper has more results, including results for Julia, Lua, and R.

    <center>
        <b>Question</b>: are there other baselines we should include?
    </center>
    """)
    return


@app.cell(hide_code=True)
def _(agnostics_ml_accuracy, mo, qwen2p5coder_ml_accuracy):
    mo.md(rf"""
    ## Comparisons for this Talk

    **Model:** Qwen Coder 2.5 3B Instruct

    **Benchmark:** The language-agnostic version of HumanEval, prefixed with "Write an OCaml program to solve this problem:".

    Model                    | Result
    -------------------------|-------------------------------
    Baseline: original model | {qwen2p5coder_ml_accuracy:.2f}
    Agnostics trained        | {agnostics_ml_accuracy:.2f}
    ...                      |

    *Are there other baselines we should include?*

    **Note:** GPT-5 Mini gets 0.72 on this benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(
    Optional,
    StandardModel,
    batches,
    bounded_subprocess_async,
    litellm,
    run_bounded,
    tempfile,
    torch,
    tqdm,
):
    """
    The code in this cell is based on the evaluation methodology of Agnostics (https://arxiv.org/abs/2508.04865).
    Instead of language-specific tests and tasks, we test programs on their I/O behavior. 
    """

    async def predictions_litellm(
        *,
        model: str,
        test_set: list,
        batch_size: int = 50,
        **extra_args,
        ):
        """
        See _predictions_1batch for the expected format of test_set. The model should
        be a litellm model string, e.g., "openai/..." or "anthropic/..." The batch_size
        is the number of concurrent requests to make.

        Any extra keyword arguments are passed to litellm.acompletion.
        """
        pbar = tqdm(desc="Predictions", total=len(test_set))

        async def do(item):
            try:
                resp = await litellm.acompletion(
                    model,
                    messages=[{"role": "user", "content": item["task"]}],
                    **extra_args,
                )
            except Exception as e:
                return [ { **item, "prediction": "", "error": str(e) } ]

            return [ { **item, "prediction": choice.message.content } for choice in resp.choices ]

        async def tasks():
            for item in test_set:
                pbar.update(1)
                yield do(item)

        return [
            item for items in [ await r async for r in run_bounded(tasks(), limit=batch_size) ]
                 for item in items
        ]


    def _predictions_1batch(*, model: StandardModel, test_set, max_tokens: int):
        """
        Produces standard chat completions for every item, using item["task"] as the
        prompt. It returns a list of updated items, where each item has a new field 
        called `"prediction"`.
        """
        prompts = [ item["task"] for item in test_set ]
        input_ids = model.chat_tokenize(prompts)

        with torch.no_grad(), model.lm.generate(input_ids, max_new_tokens=max_tokens, do_sample=False):
              outputs = model.lm.generator.output.save()
        # Omit the prompt
        outputs = outputs[:, input_ids.shape[1]:]
        decoded_outputs = model.lm.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [
            { **item, "prediction": decoded_outputs[i] }
            for i, item in enumerate(test_set)
        ]


    def predictions(*, model: StandardModel, test_set: list, max_tokens: int, batch_size: int):
        """
        Applies _predictions_1batch in batches.
        """
        return [
            item
            for batch in tqdm(list(batches(test_set, batch_size=batch_size, epochs=1)))
            for item in _predictions_1batch(model=model, test_set=batch, max_tokens=max_tokens)  
        ]

    def extract_code_from_markdown(markdown: str) -> Optional[str]:
        """
        Extracts the *first* markdown code block from a  markdown document.
        """
        code_block_start = markdown.find("```")
        if code_block_start == -1:
            return None

        # Skip past the opening ```
        code_start = code_block_start + 3

        code_block_end = markdown.find("```", code_start)
        if code_block_end == -1:
            return None

        code = markdown[code_start:code_block_end].strip()

        # There have been times we have used this, but I believe it is a hack.
        # if "# Example usage:" in code:
        #     code = code.split("# Example usage:")[0]

        # Remove language tag from the first line, if present.
        first_newline = code.find('\n')
        if first_newline > 0:
            # Consider the case where the block begins with "```python\n...". In this
            # case, code would already be "python\n..." and first_newline would be 7.
            # Thus first_newline + 1 is the index of "...".
            code = code[first_newline + 1:]

        return code.strip()

    async def _test_io(*, suffix: str, command: list, prediction: dict) -> dict:
        """
        Runs the I/O tests on a predicted program. The prediction must have the following fields:

        ```python
        {
            "prediction": str # text output from the LLM, with OCaml in a markdown block
            "io_examples": str # I/O outputs in the Agnostics HumanEval format
        }
        ```

        The function will extend `prediction` with these fields:

        ```python
        {
            "success": bool,
            "extracted_code": Optional[str] # None if it fails to extract code
            "stderr": Optional[str] # standard error output on the test that failed, if any
        }    

        We write each predicted program to a temporary file (with the extension `suffix`). Suppose
        this file is `tmp.suffix`, then the program `[*command, "tmp.suffix"]` should compile and
        run the program.
        """
        code = extract_code_from_markdown(prediction["prediction"])
        prediction = { **prediction, "extracted_code": code, "stderr": None }
        if code is None:
            return { **prediction, "success": False }
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix) as f:
            f.write(code)
            f.flush()
            for (stdin_text, stdout_text) in prediction["io_examples"]:
                result = await bounded_subprocess_async.run(
                    args=[*command, f.name],
                    stdin_data=stdin_text
                )
                if result.exit_code != 0:
                    return { **prediction, "success": False, "stderr": result.stderr }
                if result.stdout.strip() != stdout_text.strip():
                    return { **prediction, "success": False, "stderr": result.stderr }
        return { **prediction, "success": True }

    async def executions(*, suffix: str, command: list, predictions: list, num_threads: int):
        """
        Runs all I/O tests on predictions. See _test_io for more information.
        """
        pbar = tqdm(desc="Executions", total=len(predictions))

        async def tasks():
            for prediction in predictions:
                pbar.update(1)
                yield _test_io(suffix=suffix, command=command, prediction=prediction)

        executions = [ await r async for r in run_bounded(tasks(), limit=num_threads) ]
        return (
            sum(1 for item in executions if item["success"]) / len(executions),
            executions
        )
    return executions, predictions, predictions_litellm


@app.cell
async def _(disk_cache, executions, ml_test_set, predictions_litellm):
    # To run this cell without the cached results, first start the model locally:
    #     uvx vllm serve qwen2p5_coder_3b_instruct_agnostics_cf
    with disk_cache("cache/humaneval-ml-qwen2p5_coder_3b_instruct_agnostics_cf.pkl"):
        agnostics_ml_predictions = await predictions_litellm(
            model="openai/qwen2p5_coder_3b_instruct_agnostics_cf",
            test_set=ml_test_set,
            batch_size=200,
            temperature=0.0,
            max_tokens=1024,
            base_url="http://localhost:8000/v1",
            api_key="ignored")
        (agnostics_ml_accuracy, agnostics_ml_executions) = await executions(
            suffix=".ml",
            command=["ocaml"],
            predictions=agnostics_ml_predictions,
            num_threads=50)
    return (agnostics_ml_accuracy,)


@app.cell
async def _(disk_cache, executions, ml_test_set, predictions_litellm):
    # To run this cell without the cached results, first start the model locally:
    #     uvx vllm serve ~/models/qwen2p5coder_3b_instruct --served-model-name qwen2p5coder_3b_instruct
    # Shut down the model when done, because we need a different model for the next cell.
    with disk_cache("cache/humaneval-ml-qwen2p5_coder_3b_instruct.pkl"):
        qwen2p5coder_ml_predictions = await predictions_litellm(
            model="openai/qwen2p5coder_3b_instruct",
            test_set=ml_test_set,
            batch_size=200,
            temperature=0.0,
            # These problems can all be solved in far less than 1K tokens. Any response
            # that is this long is surely degenerate.
            max_tokens=1024,
            base_url="http://localhost:8000/v1",
            api_key="ignored")
        (qwen2p5coder_ml_accuracy, qwen2p5coder_ml_executions) = await executions(
            suffix=".ml",
            command=["ocaml"],
            predictions=qwen2p5coder_ml_predictions,
            num_threads=50)
    return qwen2p5coder_ml_accuracy, qwen2p5coder_ml_executions


@app.cell
async def _(
    agnostic_test_set,
    disk_cache,
    executions,
    ml_test_set,
    predictions_litellm,
):
    _test_set = [
        { **item, "task": f"Write an OCaml program to solve this problem (use only builtins and do not use the Base or Core libraries): {item["task"]}" }
        for item in agnostic_test_set
    ]
    with disk_cache("cache/humaneval-ml-qwen2p5_coder_3b_instruct_prompt-optimized.pkl"):
        qwen2p5coder_promptoptimized_ml_predictions = await predictions_litellm(
            model="openai/qwen2p5coder_3b_instruct",
            test_set=ml_test_set,
            batch_size=200,
            temperature=0.0,
            # These problems can all be solved in far less than 1K tokens. Any response
            # that is this long is surely degenerate.
            max_tokens=1024,
            base_url="http://localhost:8000/v1",
            api_key="ignored")
        (qwen2p5coder_promptoptimized_ml_accuracy, qwen2p5coder_promptoptimized_ml_executions) = await executions(
            suffix=".ml",
            command=["ocaml"],
            predictions=_predictions,
            num_threads=50)
    return (qwen2p5coder_promptoptimized_ml_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What's Going on Inside These Models?
    """)
    return


@app.cell
def _(agnostic_test_set):
    ml_test_set = [
        { **item, "task": f"Write an OCaml program to solve this problem: {item["task"]}" }
        for item in agnostic_test_set
    ]

    py_test_set = [
        { **item, "task": f"Write a Python program to solve this problem: {item["task"]}" }
        for item in agnostic_test_set
    ]
    return ml_test_set, py_test_set


@app.cell
def _(agnostic_test_set):
    _max_len = 120
    _max_rows = 5
    _tasks = [ item["task"][:_max_len] for item in agnostic_test_set ]
    text_rows = [ f"<tr><td>{x} ...</td></tr>" for x in _tasks ][:_max_rows]
    return (text_rows,)


@app.cell(hide_code=True)
def _(mo, model, text_rows):
    mo.md(rf"""
    ## The Model and Datasets

    1. Model: Qwen 2.5 Coder 3B Instruct. *A really good Code LLM for its size class.*

    2. Dataset: 151 task from prompts from Agnostic-HumanEval, e.g., 

    <table>
        {''.join(text_rows)}
    </table>

    3. We prefix "Write an OCaml program to solve this problem:" or "Write a Python program to solve this problem:" to every prompt.

    **Note:** This is a chat / instruction model, and the input is first templatized with chat tokens:

    For the input prompt `"Write hello world in OCaml."`, the model actually receives:

    ```
    {model.lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write hello world in OCaml."}],
        tokenize=False,
        add_generation_prompt=True)}
    ```

    What it really receives are input tokens, and there are *five* tokens that appear after the task prompt:

    ```
    {model.lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write hello world in OCaml."}],
        tokenize=True,
        add_generation_prompt=True)}
    ```
    """)
    return


@app.cell
def _(mo, model):
    mo.output.append(mo.md("## The Last Five Tokens of a Prompt"))

    _tokenizer = model.lm.tokenizer

    _tokens = _tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write hello world in OCaml."}],
        tokenize=True,
        add_generation_prompt=True)[-5:]

    mo.output.append(mo.md(f"Tokens: `{_tokens}`"))
    mo.output.append(mo.md(f"Decoded Tokens: `{repr([_tokenizer.decode(x) for x in _tokens])}`"))
    mo.output.append(mo.md("""We are *not* going to do the usual thing, which is to ask "what's the next predicted token"?"""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## A High-Level Schematic of a Decoder-Only, Dense Language Model

    {mo.image("public/transformer.jpg", width="50%")}
    """)
    return


@app.cell
def _():
    r2048 = """\\mathbb{R}^{2048}"""
    return (r2048,)


@app.cell(hide_code=True)
def _(mo, model, r2048):
    mo.md(rf"""
    ## Our Model in Code

    The model definition for Qwen 2.5 Coder 3B Instruct:

    ```
    {repr(model.lm)}
    ```

    Three major parts (with simplifications):

    - The embedding layer (`model.embed_tokens`) that maps integer tokens to vectors $v \in {r2048}$.
    - Each decoder layer (`model.layers[0] ... model.layers[35]`) transforms its input to an
      an output vector $v \in {r2048}$
    - The language modelling head (`model.lm_head`) maps the output from `model.layers[35]` to a distribution
      over output tokens

    We will examine `model.layer[i].output` for every layer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NNSight and NDIF

    Fiotto-Kaufman, et al. [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://openreview.net/forum?id=MxbEiFRf39). *ICLR*, 2025


    <span>
    <img src="https://ndif.us/images//NDIF_Acr_color.png" width="30%">
    </span>

    - We can use NNSight and the [National Deep Inference Fabric](https://ndif.us) to access and manipulate model internals
    - NDIF hosts very large models (e.g., Llama 3 405B) for interpretability research on National Science Foundation's Delta Cluster.
    - NNSight is a Python DSL for inspecting and manipulating model internals: it supports both local models and models hosted on NDIF
    """)
    return


@app.cell
def _(LanguageModel, Path, StandardModel, torch):
    lm  = LanguageModel(
        "/mnt/ssd/arjun/models/qwen2p5coder_3b_instruct" if Path("/mnt/ssd/arjun/models").exists() else "Qwen/Qwen2.5-Coder-3B-Instruct", 
        dtype=torch.bfloat16,
        dispatch=torch.cuda.is_available(),
        device_map="cuda" if torch.cuda.is_available() else "mps")
    model = StandardModel(lm, {})
    return lm, model


@app.cell
def _(lm, mo, torch):
    _txt = "Write a program to solve this problem: fizz buzz"
    # Apply the chat template
    _input_ids = lm.tokenizer.apply_chat_template(
        [ { "role": "user", "content": _txt } ],
        add_generation_prompt=True, 
        return_tensors="pt")
    # Save the intermediate values / residuals with NNSight
    if torch.cuda.is_available():
        with lm.forward(_input_ids):
            intermediate_values = [ ]
            for layer in lm.model.layers:
                intermediate_values.append(layer.output[0, -5:])
                #                                       │   └──── last five token positions
                #                                       └──────── first (and only) prompt in the batch
            intermediate_values.save()
        result = intermediate_values[0].shape
    else:
        # I bought a MacBook Air with 8B RAM
        result = None
    
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo, r2048):
    mo.md(rf"""
    ## Looking at Values in ${r2048}$

    1. We will project to two-dimensions using *principal components analysis* (PCA). PCA is a two step procedure:
       - From a dataset, where each item is a point in ${r2048}$, we do change of basis from the standard basis vectors
         $(1, 0, \cdots), (0, 1, 0, \cdots), \cdots$ to to a new list of basis vectors that ranked by how much variation they capture in the data.
       - We pick the top $N=2$ to get the $N$ most significant directions of variance

    2. We apply PCA to the last five tokens of our task prompts (and not the programs) at every layer.

    3. **Dataset**: HumanEval programming tasks prefixed with "Write a Python program..." and "Write an OCaml program ..."


    *Work with Md Mahbubur Rahman (Iowa State) and Harshita Menon (Lawrence Livermore National Lab).*
    """)
    return


@app.cell
def _(
    disk_cache,
    itertools,
    last_token_residuals,
    ml_test_set,
    mo,
    model,
    pd,
    py_test_set,
):
    _num_residuals = 5

    _all_tasks = [ item["task"] for item in itertools.chain(py_test_set, ml_test_set)]
    _input_ids = model.chat_tokenize(_all_tasks)
    residual_df = pd.DataFrame({
            "task": _all_tasks,
            "label": ["Python"] * len(py_test_set) + ["OCaml"] * len(ml_test_set)
        })
    with disk_cache("cache/py_ml_residuals.pkl"):
        py_ml_residuals = last_token_residuals(model, _input_ids, n=_num_residuals, batch_size=50)
    token_pos_slider = mo.ui.slider(0, _num_residuals - 1)
    layer_slider = mo.ui.slider(0, len(model.layers) - 1)
    return layer_slider, py_ml_residuals, residual_df, token_pos_slider


@app.cell
def _(
    layer_slider,
    mo,
    pca_plot,
    py_ml_residuals,
    residual_df,
    token_pos_slider,
):
    residual_df["residual"] =  [z for z in py_ml_residuals[layer_slider.value].to(float).numpy()[:, token_pos_slider.value, :]]
    mo.vstack((
        mo.md("## Visualizing Intermediate Values"),
        "Token Position (from end):",
        token_pos_slider,
        "Layer:",
        layer_slider,
        pca_plot(residual_df, "residual", "label", ["label", "task"])))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Activation Steering

    [Ravfogel et al., 2021](https://aclanthology.org/2021.conll-1.15), [Rimsky et al., 2024](https://aclanthology.org/2024.acl-long.828/), [Li et al., 2024](https://openreview.net/forum?id=aLLuYpn83y)
    """)
    return


@app.cell
def _(mo, py_ml_residuals, py_test_set):
    _num_tasks = len(py_test_set)
    assert py_ml_residuals[0].shape[0] == _num_tasks * 2

    # Two datasets of intermediate values by layer.
    # Each list element has shape (_num_tasks, num_positions, hidden_size)
    # where num_positions == 5 and hidden_size is 2048
    _py_intermediate_values = [ a[:_num_tasks, :, :] for a in py_ml_residuals ]
    _ml_intermediate_values = [ a[_num_tasks:, :, :] for a in py_ml_residuals ]

    # Mean intermediate values
    _mean_py = [ a.mean(dim=0) for a in _py_intermediate_values ]
    _mean_ml = [ a.mean(dim=0) for a in _ml_intermediate_values ]

    # Difference of means -- start at Python and go toward OCaml
    lang_changing_patch = [ -py + ml for (ml, py) in zip(_mean_ml, _mean_py) ]
    mo.show_code()
    return (lang_changing_patch,)


@app.cell
def _(List, mo, torch):
    @torch.no_grad()
    def generate_with_patch(model, inputs: List[str | torch.Tensor], patches: List[torch.Tensor], max_new_tokens: int) -> List[str]:
        """
        Generate text from model with patches applied to several intermediate values of
        the prompt.

        Patches is a list of tensors where patches[i].shape = (j, hidden_size). We interpret
        patch[i][j] as the patch for layer i at prompt position -j. Thus patches[0][-1] is
        the patch for the last prompt token at layer 0.
        """
        device = model.lm.device

        with model.lm.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False):
            layers = model.layers
            # Patch every layer. Can probably do fewer if desired. If the list is
            # empty, we just generate text normally.
            for i, layer_patches in enumerate(patches):
                # layer_patches[-1], layer_patches[-2], ... are the patches for the
                # last token, second last token, etc. patch_start_ix is the offset
                # where we start patching.
                patch_start_ix = inputs.shape[1] - layer_patches.shape[0]
                for j, patch in enumerate(torch.unbind(layer_patches)):
                    # We cannot use -j as the index since we are generating new tokens
                    # k is effectively -j if the sequence were just the prompt.
                    k = patch_start_ix + j
                    patch = patch.unsqueeze(0).to(device)

                    # Patch the output
                    layers[i].output[:, k, :] = layers[i].output[:, k, :] + patch

            outputs = model.lm.generator.output.save()
        # Decode the output tokens, skip the prompt and padding
        return model.lm.tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)

    mo.show_code()
    return (generate_with_patch,)


@app.cell
def _(
    agnostic_test_set,
    disk_cache,
    generate_with_patch,
    lang_changing_patch,
    mo,
    model,
):
    _max_new_tokens = 150
    _prompts = [ item["task"] for item in agnostic_test_set[0:3] ]
    with disk_cache("cache/steered_output_text.pkl"):
        steered_output_results = generate_with_patch(
            model,
            inputs=model.chat_tokenize(_prompts),
            patches=[ x for x in lang_changing_patch ],
            max_new_tokens=_max_new_tokens)
    mo.output.append(mo.md(f"## Patching Results: Default Language Change\n\nOutput clipped to {_max_new_tokens} output tokens."))
    for _prompt, _completion in zip(_prompts, steered_output_results):
        mo.output.append(mo.md("**Prompt**"))
        mo.output.append(mo.plain_text(_prompt))
        mo.output.append(mo.md("**Completion**"))
        mo.output.append(mo.plain_text(_completion))
        mo.output.append(mo.md("---"))
    return


@app.cell
def _(
    agnostic_test_set,
    disk_cache,
    generate_with_patch,
    lang_changing_patch,
    mo,
    model,
    torch,
):
    _max_new_tokens = 50
    _prompts = [ item["task"] for item in agnostic_test_set[0:3] ]
    with disk_cache("cache/random_steered_output_text.pkl"):
        random_steered_results = generate_with_patch(
            model,
            inputs=model.chat_tokenize(_prompts),
            patches=[ torch.rand_like(x) for x in lang_changing_patch ],
            max_new_tokens=_max_new_tokens)
    mo.output.append(mo.md(f"## Patching Baseline: Random Patches\n\nOutput clipped to {_max_new_tokens} output tokens."))
    for _prompt, _completion in zip(_prompts, random_steered_results):
        mo.output.append(mo.md("**Prompt**"))
        mo.output.append(mo.plain_text(_prompt))
        mo.output.append(mo.md("**Completion**"))
        mo.output.append(mo.plain_text(_completion))
        mo.output.append(mo.md("---"))
    return


@app.cell(hide_code=True)
def _(
    agnostics_ml_accuracy,
    bug_fixing_steering_accuracy,
    mo,
    qwen2p5coder_ml_accuracy,
    qwen2p5coder_promptoptimized_ml_accuracy,
):
    mo.md(rf"""
    # Back to Fine-tuned Models: Steering Away from Bugs in OCaml

    - Dataset: *only* the OCaml tasks -- just the prompts and not the solutions!
    - Partition: two sets, based on whether or not the original model can solve the task

    Model                    | Result
    -------------------------|-------------------------------
    Baseline: original model | {qwen2p5coder_ml_accuracy:.2f}
    Agnostics trained        | {agnostics_ml_accuracy:.2f}
    Steering away from bugs  | {bug_fixing_steering_accuracy:.2f}
    Hand-optimized prompt    | {qwen2p5coder_promptoptimized_ml_accuracy:.2f}

    **My conclusion:** On this model, the RL effort was worthwhile, but the gap to baselines is narrowed.
    """)
    return


@app.cell
async def _(
    StandardModel,
    batches,
    disk_cache,
    executions,
    generate_with_patch,
    itertools,
    last_token_residuals,
    model,
    qwen2p5coder_ml_executions,
    torch,
    tqdm,
):
    @torch.no_grad()
    def patched_completions2(
        model: StandardModel,
        *,
        patches,
        test_set,
        max_tokens: int,
        ):

        prompts = [ item["task"] for item in test_set ]
        input_ids = model.chat_tokenize(prompts)
        decoded_outputs = generate_with_patch(model=model, inputs=input_ids, patches=patches, max_new_tokens=max_tokens)
        return [
            { **item, "prediction": decoded_outputs[i] }
            for i, item in enumerate(test_set)
        ]


    _ok_items = [ item for item in qwen2p5coder_ml_executions if item["success"] ]
    _err_items = [ item for item in qwen2p5coder_ml_executions if not item["success"] ]

    # Two datasets of intermediate values by layer.
    # Each list element has shape (_num_tasks, num_positions, hidden_size)
    # where num_positions == 5 and hidden_size is 2048
    _num_residuals = 5
    _ok_tasks = [ item["task"] for item in _ok_items ]
    _err_tasks = [ item["task"]  for item in _err_items ]
    with disk_cache("cache/steering_away_from_bugs.pkl"):
        ok_intermediate_values = last_token_residuals(
            model, model.chat_tokenize(_ok_tasks), n=_num_residuals, batch_size=50)
        err_intermediate_values = last_token_residuals(
            model, model.chat_tokenize(_err_tasks), n=_num_residuals, batch_size=50)


        # Mean intermediate values
        mean_ok = [ a.mean(dim=0) for a in ok_intermediate_values ]
        mean_err = [ a.mean(dim=0) for a in err_intermediate_values ]

        # Difference of means -- move away from errors and towards OK
        bug_fixing_patch = [ -err + ok for (ok, err) in zip(mean_ok, mean_err) ]

        # Steer the failures toward the OKs
        bug_fixing_predictions = [
            item
            for batch in tqdm(list(batches(_err_items, batch_size=50, epochs=1)))
            for item in patched_completions2(model, patches=bug_fixing_patch, test_set=batch, max_tokens=512)  
        ]
        (_, bug_fixing_executions) = await executions(
            suffix=".ml",
            command=["ocaml"],
            predictions=bug_fixing_predictions,
            num_threads=50)

    bug_fixing_steering_accuracy = sum(
        item["success"] 
        for item in itertools.chain(bug_fixing_executions, _ok_items)) / len(qwen2p5coder_ml_executions)
    return (bug_fixing_steering_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Understanding Type Mispredictions with Activation Steering

    How do Code LLMs (mis) predict type annotations? E.g.,

    ```python
    def func(n: [FILL]):
        n = n.lower()
        return n[::-1] == n
    ```

    Do they use lexical features (e..g, *variables names s are usually strings*) or do they actually understand the types in the program?

    We study *type annotation prediction* for Python and TypeScript:

    - Results are much clearer when models are good at a task (i.e., results would likely be messy with OCaml)

    - By type annotation prediction we mean, "Does the model predict the type annotation the programmer had written down?" which is *not* the same as "Does the model predict any type annotation that type-checks" (which could be the *any* type.)

    - *Positive Dataset 1:* TypeScript programs from The Stack ([Kocetkov et al., 2022](https://huggingface.co/papers/2211.15533) that type check

    - *Positive Dataset 2:* Python programs that have many type annotations from ManyTypes4Py ([Mir et al., 2021](https://ieeexplore.ieee.org/document/9463150)) that type check with PyRight

    - *Negative Dataset:* Semantics-preserving edits to positive examples until the LLM mispredicts

    {mo.image("public/mutation_operators.jpg", width="50%")}

    *From:* Francesca Lucchetti and Arjun Guha. [Understanding How CodeLLMs (Mis)Predict Types with Activation Steering](https://aclanthology.org/2025.blackboxnlp-1.22/). *BlackboxNLP*, 2025
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Understanding Type Mispredictions with Activation Steering

    1. We can correct a significant portion of type mispredictions across several models:

    {mo.image("public/steering_types_for_ts_all_models.jpg", width="50%")}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Steering Does Not Increase Type Precision

    2. Steering *corrects mispredictions that are type errors*:

    {mo.image("public/type_steering_fixes_type_errors.jpg", width="50%")}

    - *x*-axis: does the model mispredict a type is actually OK. E.g., predicting *any* is usually OK, even if
       it is a misprediction
    - *y*-axis: accuracy of steering

    1. Steering accuracy is high when the model predicts a type that leads to a type error
    2. Steering accuracy is low when the model predicts a type that passes, but is technically wrong, e.g.
       predicts *any* when there is a better type to predict

    *We are steering away from type errors, and not steering to increase type precision.*

    *From:* Francesca Lucchetti and Arjun Guha. [Understanding How CodeLLMs (Mis)Predict Types with Activation Steering](https://aclanthology.org/2025.blackboxnlp-1.22/). *BlackboxNLP*, 2025
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Evidence of a Shared Direction that Encodes Types for Both TypeScript and Python

    {mo.image("public/steering_transfer.jpg", width="50%")}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Wrapping Up

    - We have a surface-level understanding of what LLMs can and cannot do, and are just
      beginning to dig deeper with interpretability techniques
    - Formal properties of code -- ability to mutate, test, type-check, etc. -- make programming
      languages a good platform for studying model internals

    **However:**

    - *What tasks can an LLM do?*: is a narrow question
    - We also need to understand humans' mental models of LLM capabilities to truly understand
      what is possible with LLMs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompts and People
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## You Want To Do: Can A Model Do Task *X*?

    **A benchmark measures:** "Can a model produce the correct answer to this prompt *P*?"

    A prompt from ParEval ([Nichols et al., 2024](https://dl.acm.org/doi/10.1145/3625549.3658689))

    ```cpp
    /* Compute the prefix sum of the vector x into output.
       Use OpenMP to compute in parallel.
       Example:

       input: [1, 7, 4, 6, 6, 2]
       output: [1, 8, 12, 18, 24, 26]
    */
    void prefixSum(std::vector<double> const& x, std::vector<double> &output)
    ```

    Maybe it just needs a little more detail?

    Adding detailed (LLM-generated) instructions to prompts can significantly boost performance on benchmarks.

    {mo.image("public/partialordereval.png", width="30%")}

    Models are Qwen 2.5 Coder Instruct (14B,7B,3B,1.B)

    *Source:* Yangtian Zi, Harshita Menon, and Arjun Guha. More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation. *AACL*, 2025
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Can A Person Prompt a Model To Do A Task?

    1. Lots of research on student-LLM interactions
       - Based on paper titles, 20% of the papers at SIGCSE 2025 are about LLMs and CS education (per Sonnet)
       - I am part of the problem: have been studying student-LLM interactions since 2023
    2. Our user studies lend themselves to interesting secondary analyses

    We conducted an experiment in **January--March 2023** with 120 students who had completed CS1 in Python (and no other course):

    - ChatGPT had just been released (December 2022)
    - None of them had used GitHub Copilot
    - None of them had used ChatGPT
    - Most had heard of neither ChatGPT nor Copilot

    High-level question: how do students with **zero LLM experience** but **basic programming knowledge** do at prompting state-of-the-art Code LLM (from 2023)?

    *Source:* Hannah McLean Babe, Sydney Nguyen, Yangtian Zi, Arjun Guha, Carolyn Jane Anderson, and Molly Q Feldman. [How Beginning Programmers and Code LLMs (Mis)read Each Other](https://dl.acm.org/doi/full/10.1145/3613904.3642706). *CHI*, 2024.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Studying Student Prompting Ability

    1. **Model**: code-davinci-002 (easily best Code LLM from early 2023)
    2. **Population**: 120 students from Wellesley, Oberlin, and Northeastern who have 1 semester of CS1 in Python
    3. **Problems**: trivial programming problems taken from their homeworks and exams of their CS1 class.
    > Write a function that takes a list of numbers and an exponent, and returns the numbered exponentiated.
    4. **Task**: Only writing a natural language prompt
       - Cannot edit code. If the LLM produces wrong code, they have to revise the prompt (or roll the dice again)
       - No need to determine correctness: we run unit tests and report full results.
    5. **Format**: Zoom study 1-on-1 with a researcher; 60 minutes to do six tasks; can retry as many times as desired, or give up

    {mo.image("public/charlie.png", width="50%")}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.output.append(mo.md("## Basic Results"))
    mo.output.append(mo.image("public/charlie_success_rates.png", width="50%"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Dataset

    - We publish an anonymized dataset of 2,000+ student written prompts
    - The dataset has the full "prompt trajectory" for each student, generated code, test results, etc.

    - StudentEval: we turn the first and last prompts from each trajectory into a benchmark
      with *several prompts of varying quality per task*. Most benchmarks do not do this.

    - Shows how models do on prompts that originally failed or succeeded
    - *First success* prompts are typically easier than *last success*

    {mo.image("public/studenteval.png", width="40%")}

    *Source:* Hannah McLean Babe, Sydney Nguyen, Yangtian Zi, Arjun Guha, Molly Q Feldman, and Carolyn Jane Anderson. [StudentEval: A Benchmark of Student-Written Prompts for Large Language Models of Code](https://aclanthology.org/2024.findings-acl.501/). *ACL Findings*, 2024
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What Makes Student-Written Prompts Unreliable?

    **Example Problem (what a student sees)**

    ```python
    # Signature
    def total_bill(grocery_list, sales_tax):
        ...

    # Examples
    total_bill([['eggs', 6, 0.99], ['milk', 1, 1.49], ['bread', 2, 3.5]], 0.07) # 15.44
    total_bill([['eggs', 6, 0.99], ['milk', 1, 1.49], ['bread', 2, 3.50]], 0.0) # 14.43
    total_bill([['bread', 2, 3.50]], 0.5) # 10.5
    ```

    **Prompt Dataset:**
    - 13 students writing very different natural language prompts for this problem
    - Some eventually succeed, some eventually fail, nobody solves it in one shot

    **Prompt Abstraction:**
    - What set set of facts -- we call them "clues" -- get added, removed, or updated on each attempt?
    - For `total_bill`: the clues are (1) First input is a list, (2) List structure explained, (3) Second input is sales tax,
      (4) Multiply item price by quantity, (5) Sum results, (6) Apply sales tax, (7) Round to two decimal places, and
      (8) Return total
    - We label every edit by the delta to the clue set

    *Source*: Francesca Lucchetti, Zixuan Wu, Arjun Guha, Molly Q Feldman, and Carolyn Jane Anderson. [Substance Beats Style: Why Beginning Students Fail to Code with LLMs](https://aclanthology.org/2025.naacl-long.433/). *NAACL*, 2025
    """)
    return


@app.cell
def _(mo):
    mo.output.append(mo.md("## Prompt Editing Trajectories for total_bill"))

    mo.output.append(mo.md("""
    - Diamond: student observes the problem/output and writes a prompt"
    - Circle: LLM produces a response, platform runs tests, and presents results to student"""))

    mo.output.append(mo.image("public/big_clues.png", width="50%", height="50%"))
    mo.output.append(mo.md("**A takeaway**: students do not have a good mental model of what the LLM already knows"))
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ## Conclusion

    1. Everyone knows someone who claims that LLMs make them 3-4x more productive.
    2. Not clear what benefit a randomly sampled developer gets from LLMs.
    3. In controlled studies, LLMs *slow down* experienced developers
       ([Vaithilingam et al., 2022](https://dl.acm.org/doi/abs/10.1145/3491101.3519665), [Becker et al., 2025](https://arxiv.org/abs/2507.09089)).
       - The 2025 METR study includes NNSight / NDIF compiler/machine learning/HPC hackers 🥲
    5. We are starting to study LLM agent use in the wild.

    {mo.image("public/agentpack.png", width="50%")}

    *Source:* Yangtian Zi, Zixuan Wu, Aleksander Boruch-Gruszecki, Jonathan Bell, Arjun Guha. [AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans](https://arxiv.org/abs/2509.21891), 2025
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extras
    """)
    return


@app.cell
def _():
    """
    ## Talk Outline

    1. *Introduction*: my background is PL research; last few years my research group has been pursuing three threads
       of research (1) LLMs for low-resource PLs, (2) studying how people use LLMs, and (3) understanding how LLMs perform
       programming tasks under the hood. I'm going to try to tie these together.


    2. *Prelude:* concrete tasks for evaluation.
       - Easy to think that models can do anything or nothing -- based on your preconceived notions
       - I'm sure people here have tried to get models to do OCaml tasks and its hard.
       - Introduce work that we've done on benchmarks for programming tasks, and use these benchmarks to guide
         the rest of the talk.
    3. *Rewind to summer 2022:* (1) 20 years ago in machine learning and dog years. We will quickly catch up to the present,
       (2) This is after Github Copilot release, but before ChatGPT
    5. State of models and benchmarking from 2021-2022
    11. *Agnostics approach to creating problems:* With newer models: ask them to synthesize whole programs,
        give them very specific formatting instructions, test input output behavior of the program instead of having
        unit tests. So, very easy to make language neutral.
    12. *Agnostics on HumanEval:* show a sample. Show model results.
    13. *Agnostics on LiveCodeBench*: present paper results.
    14. *Agnostics training:* Can apply reinforcement learning (show small models)
    15. *Looking at smaller models:* show results
    16. *Main topic:* Are these good results? What should we compare to? They are good compared to the original model.
        But, I want to consider some points of comparison.
    17. *Prompt optimization:* **TODO** use GEPA. Show my manual optimization

    *Activation steering:*

    1. How do these models process these tasks? How do they process any task?
    2. Very high-level model architecture : transformer blocks and residual streams. Output distribution of tokens.
       Articulate that we sample and predict the next token to extend the prompt.
    4. We can look at what's happening in the residual stream -- 3K dimensional vector
    5. Explain PCA: learns a change of basis in the high-dimension space. The new basis or axes are ordered by
       variance -- first dimension has the most variance, the last dimension has the least variance. The
       When we project to 2D, we pick the the first and second axes and drop the rest. So, a linear separation
       that appears in 2D just may be -- but not guaranteed  -- to be a linear separation in higher-dimension.
    6. *Do PCA on the OCaml and Python prompts*
    7. Activation steering to change language: just do it to change language first.
    8. Activation steering to fix bugs:
       - Look at the prompts the model gets wrong -- can we just steer them?
    9. Francesca's paper results (a couple of slides)
    10. There are lots of ways to fine-tune models, but you have to think hard about baseline results.

    *NDIF:*

    ???

    *Studying prompts and people:*

       - What if there was more information, would the model be able to solve it? Show a graph from PartialOrderEval
    3. The prompts come from programmers, and we've been studying student programmers for some time
    4. Lots of studies of students using LLMs. Claude Sonnet estimates based on paper titles that ~20% of SIGCSE 2025 papers are about students, LLMs
    5. We have conducting a lab study that's quite different, and lends itself to depth of analysis
       a. Rewind back to Jan/Feb 2023: (20 years ago)
       b. ChatGPT had just been released. Your average CS student hadn't used it. Many hadn't heard about it. We did an experiment with 120 students who had zero GenAI experience.
       c. Jensen quote
    6. At least two things you need to do: write a prompt and check that the model output is correct.
    7. Our web-based platform: takes care of checking tests. We also use the tests to convey the intended task
       - This is tricky -- if you convey the task in prose, a student can just paste the prose in.
    8. Artifact: the StudentEval dataset. Explain that we can use it to evaluate models.
    9. In-experiment results: students struggle
    10. Substance v style results
    11. Starting to study real programmers
        - It is not clear that the a randomly sampled programmer gets much out of these tools
        - METR study suggests that experienced developers can be slowed down by AI when working in
          familiar codebases: https://arxiv.org/pdf/2507.09089
        - AgentPack

    Conclusion:
    1. How do we get models to work for programmers? Both a question of studying models and programmers.
    """
    return


@app.cell
async def _(agnostic_test_set, disk_cache, executions, predictions_litellm):
    _test_set = [
        { **item, "task": f"Write an OCaml program to solve this problem using only builtin modules (do not try to use the Base and Core libraries): {item["task"]}" }
        for item in agnostic_test_set
    ]
    with disk_cache("cache/nocore_ml_gpt5.pkl"):
        _predictions = await predictions_litellm(model="openai/gpt-5-mini", test_set=_test_set)
        _executions = await executions(suffix=".ml", command=["ocaml"], predictions=_predictions, num_threads=50)
        (gpt5_nocore_ml_accuracy, gpt5_nocore_ml_executions) = _executions
    gpt5_nocore_ml_accuracy    
    return


@app.cell
async def _(disk_cache, executions, model, predictions, py_test_set):
    with disk_cache("cache/python_baseline.pkl"):
        _py_predictions = predictions(model=model, test_set=py_test_set, max_tokens=512, batch_size=50)
        (py_accuracy, py_executions) = await executions(suffix=".py", command=["python3"], predictions=_py_predictions, num_threads=50)
    py_accuracy
    return


@app.cell
async def _(disk_cache, executions, predictions_litellm, py_test_set):
    with disk_cache("cache/python_gpt5.pkl"):
        _predictions = await predictions_litellm(model="openai/gpt-5-mini", test_set=py_test_set)
        _executions = await executions(suffix=".py", command=["python3"], predictions=_predictions, num_threads=50)
        (gpt5_py_accuracy, gpt5_py_executions) = _executions
    gpt5_py_accuracy    
    return


@app.cell
async def _(disk_cache, executions, ml_test_set, predictions_litellm):
    with disk_cache("cache/ml_gpt5.pkl"):
        _predictions = await predictions_litellm(model="openai/gpt-5-mini", test_set=ml_test_set)
        _executions = await executions(suffix=".ml", command=["ocaml"], predictions=_predictions, num_threads=50)
        (gpt5_ml_accuracy, gpt5_ml_executions) = _executions
    gpt5_ml_accuracy    
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limits of RLVR

    We should do this in depth at some point. The little result below is at a much smaller scale than the paper.
    """)
    return


@app.cell
async def _(disk_cache, executions, ml_test_set, pd, predictions_litellm):
    with disk_cache("cache/qwen2p5_coder_3b_instruct_agnostics_cf.pkl"):
        _predictions = await predictions_litellm(
            model="openai/qwen2p5_coder_3b_instruct_agnostics_cf",
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            test_set=ml_test_set,
            batch_size=50,
            temperature=0.2,
            max_tokens=1024,
            n=20,
        )
        (_, executions_ocaml_qwen2p5_coder_3b_instruct_agnostics_cf) = await executions(
            suffix=".ml", 
            command=["ocaml"],
            predictions=_predictions,
            num_threads=50
        )
    pd.DataFrame(executions_ocaml_qwen2p5_coder_3b_instruct_agnostics_cf).groupby("id")["success"].any().mean().item()
    return


@app.cell
def _():
    # with disk_cache("cache/ocaml_passk20.pkl"):
    #     _predictions = await predictions_litellm(
    #         model="openai/qwen2p5coder_3b_instruct",
    #         base_url="http://localhost:8000/v1",
    #         api_key="dummy",
    #         test_set=ml_test_set,
    #         batch_size=100,
    #         temperature=0.2,
    #         max_tokens=1024,
    #         n=20,
    #     )
    #     (_, ocaml_passk32_executions) = await executions(suffix=".ml", command=["ocaml"], predictions=_predictions, num_threads=50)
    # pd.DataFrame(ocaml_passk32_executions).groupby("id")["success"].any().mean().item()
    return


if __name__ == "__main__":
    app.run()
