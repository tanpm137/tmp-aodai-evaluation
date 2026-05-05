# tmp-aodai-evaluation

## Getting Started

1. **Environment Setup**: Create a virtual environment and install dependencies:
2. **Dataset**: Download and place the dataset in the `data/` directory.
3. **Pretrained Models**: Download and place checkpoints in the `pretrained/` directory:

   | Model Name       | Files / Components                   | Repository                                                                           |
   | :--------------- | :----------------------------------- | :----------------------------------------------------------------------------------- |
   | **BootComp**     | `bootcomp`                           | [🌐 Hugging Face](https://huggingface.co/omniousai/BootComp/tree/main)               |
   | **FitDiT**       | -                                    | [🌐 Hugging Face](https://huggingface.co/BoyuanJiang/FitDiT)                         |
   | **IMAGDressing** | -                                    | [🌐 Hugging Face](https://huggingface.co/feishen29/IMAGDressing/tree/main)           |
   | **JCo-MVTON**    | `try_on_dress.pt`, `try_on_upper.pt` | [🌐 Hugging Face](https://huggingface.co/Damo-vision/JCo-MVTON)                      |
   | **OmniTry**      | `omnitry_v1_clothes.safetensors`     | [🌐 Hugging Face](https://huggingface.co/Kunbyte/OmniTry/tree/main)                  |
   | **OOTDiffusion** | `ootd`, `humanparsing`, `openpose`   | [🌐 Hugging Face](https://huggingface.co/levihsu/OOTDiffusion/tree/main/checkpoints) |

4. **Vendors**: Original source code for benchmark methods can be found in the `vendors/` folder.
