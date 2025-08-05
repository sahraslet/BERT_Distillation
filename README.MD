In this project, we aim to reproduce the paper TinyBERT: Distilling BERT for Natural Language Understanding by Jiao et al. (2020) which focuses on knowledge distillation for transformer models. In the following we will elaborate on how we plan to implement the experiments described in the paper.

What is going to be included?
The paper introduces a two-stage distillation framework with BERT as the teacher model. The authors perform general distillation as well as task-specific distillation to achieve competitive performance with the original BERT model. They evaluate the performance of their model using the following GLUE benchmark tasks:
•	2 single-sentence tasks: CoLA, SST-2
•	3 sentence similarity tasks: MRPC, STS-B, QQP
•	4 natural language inference tasks: MNLI, QNLI, RTE
Additionally, they conducted ablation studies which were used to demonstrate the effectiveness of the learing procedure, the distillation objectives as well as the mapping function.
In this project we will focus on distilling BERT on both the pretraining and fine-tuning stage without doing the ablation studies. We will evaluate the performance of our distilled BERT on selected GLUE benchmark tasks, namely the following:
•	SST-2 for sentiment classification (single-sentence task)
•	MRPC for paraphrase detection (sentence similarity task)
•	MNLI for natural language inference (NLI task)

configs/
├── general/
│   └── tinybert_general_pretrain.yaml
├── task_specific/
│   ├── sst2_distill.yaml
│   ├── mnli_distill.yaml
│   └── rte_distill.yaml



