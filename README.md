# Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?

The repository contains code & supplement material from the paper **"Assessing LLM Text Detection in Educational
Contexts: Does
Human Contribution Affect Detection?"**.

<img src="https://anonymous.4open.science/r/Assessing-LLM-Text-Detection-in-Educational-Contexts/supplementary-material/Method.png" alt="Overview of Contribution Levels " width="500"/>

## Supplementary Material

The supplementary material for the paper, which includes dataset statistics, the actual LLM prompts, fine-tuning
details, and additional results, can be
found [here](./supplementary-material/Paper-Appendix.pdf).

## Generative Essay Detection in Education Dataset

The **G**enerative **E**ssay **D**etection in **E**ducation (**GEDE**) dataset is based on the following three text
corpora:

* Annotated Argumentative Essays [<a href="#ref1">1</a>]
* BAWE [<a href="#ref1">2</a>]
* PERSUADE 2.0 [<a href="#ref1">3</a>]

The GEDE dataset and all detector predictions from our experiments can be found in the SQLite database located in the directory `database/database.db`. For those unfamiliar with SQL databases, an CSV
version of the GEDE dataset is also provided in the `datasets/` directory.

## Code

This repository contains the full code to run all experiments. Furthermore, we include the predictions of all detectors
on the GEDE datasets.

### Environment

* Python 3.8
* PyTorch ...
* Create the environment using conda: `conda env create -f environment. yml`

### Experiments

#### Arguments

| argument             | default                             | other values                                                                                                  | description                  |
|----------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------|
| `--model`            | `detect-gpt`                        | [`fast-detect-gpt`, `intrinsic-dim`, `ghostbuster`, `roberta`, `gpt-zero`]                                    | name of the detector model   |
| `--dataset`          | `argument-annotated-essays`         | [`bawe`, `persuade`]                                                                                          | name of the data subset      |
| `--database`         | `../database/database.db`           | other path to the database                                                                                    | path to the database         |
| `--prompt_mode`      | `task`                              | [`summary`, `task+summary`, `rewrite-human`, `improve-human`, `rewrite-[job_id]`, `dipper-[job_id]`, `human`] | name of the text category    |
| `--generative_model` | `meta-llama/Llama-3.3-70B-Instruct` | `gpt-4o-mini-2024-07-18`                                                                                      | name of the generative model |

Note: In order to run `Ghostbuster`, you need to provide a valid OpenAI-API key, as this model requires access to the
OpenAI-API.

### Use our own Detector

You can evaluate your own detector by inheriting from the `Detector` class in `detectors/detector_interface.py`:

```python
from detector_interface import Detector


class MyOwnDetector(Detector):

    def __init__(self, args, **kwargs):
        super().__init__(name="MyDetector", args=args)

    def run(self, data):
        # assigning a hash to the dataset (optional)
        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])

        # Your actual detection code:

        # saving predctions to database
        # predictions is a list of floats (predictions for each sample)
        # Note: Please provide only the prediction of the sample being LLM
        self.save(predictions, answer_ids=data.id)

```

After impementing your detector, you can include it in the experiments by writing this into the `main.py` file:

```python
...
# You can place your own detector here
# ---------------------------------
elif args.model == "my-detector":
logger.info("Executing MyDetector model")
detector = MyOwnDetector(args)
# ---------------------------------
...
```

## References

<a id="ref1"></a>**[1]** Christian Stab and Iryna Gurevych. 2017. Argument Annotated Essays (version 2). https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

<a id="ref2"></a>**[2]** Hilary Nesi, Sheena Gardner, Paul Thompson, and Paul Wickens. 2008. British Academic Written English Corpus. http://hdl.handle.net/20.500.14106/2539 Literary and Linguistic
Data Service.

<a id="ref3"></a>**[3]** Scott A. Crossley, Yu Tian, Perpetual Baffour, Alex Franklin, Meg Benner, and Ulrich Boser. 2024. A large-scale corpus for assessing written argumentation: PERSUADE 2.0.
Assessing Writing 61 (2024), 100865. [https://doi.org/10.1016/j.asw.2024.100865](https://doi.org/10.1016/j.asw.2024.100865)

## Citation

Please cite this work using the following BibTex entry:

```
```
