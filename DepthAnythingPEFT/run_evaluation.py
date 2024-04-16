from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from Model import DepthAnythingPEFT
from Dataset import FlSeaDataset
import torch
import pandas as pd
from peft import LoraConfig
from Evaluation import EvaluationMetric
from torch.utils.data import Subset
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

def model_evaluation(model,image_processor,dataset_root_dir,data_use_percentage,batch_size,output_file_name):

    print("Moddel Loaded")

    data_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )

    print("Dataset Loaded")
    dataset = FlSeaDataset(root_dir= dataset_root_dir, transform=data_transforms)
    useful_dataset_length = int(len(dataset) * data_use_percentage /100)
    useful_dataset = Subset(dataset,list(range(useful_dataset_length)))
    dataloder = torch.utils.data.DataLoader(useful_dataset, batch_size= batch_size, shuffle=True)

    model.eval()
    evaluation = EvaluationMetric(False)
    all_metrics = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        print("evaluation started...")
        for i, (inputs, labels) in enumerate(dataloder):
            inputs, labels = inputs.to(device),labels.to(device)
            outputs = model(inputs).predicted_depth
            metrics = evaluation.compute_metrics(inputs,outputs,labels)
            print(metrics)
            all_metrics.append(metrics)
            print("evaluation num",i)

    print("evaluation ended; processing data...")
    # Create a DataFrame from the list of evaluation metrics
    df = pd.DataFrame(all_metrics, columns=['absrel', 'silog', 'pearson_corr', 'psnr', 'ssim'])

    # Add a row for the sum of all metrics
    mean_row = df.mean()
    mean_row.name = 'Mean'
    df = pd.concat([df, pd.DataFrame([mean_row])])

    # Save the DataFrame as a CSV file

    df.to_csv(output_file_name)

    print("Evaluation metrics saved as", output_file_name)


def main():

    # Parameters
    MODEL_CHECKPOINT = "LiheYoung/depth-anything-small-hf"
    TRAINED_CHECKPOINT = "DepthAnything/scripts/model/depth-anything-small-hf_7.pth"
    IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    DATASET_ROOT_DIR = "/mundus/abhowmik697/FLSea_Dataset"
    DATA_USE_PERCENTAGE = 100
    BATCH_SIZE = 16
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.001
    BIAS = "lora_only"

    # Model Loading
    depth_anything = DepthAnythingPEFT(model_checkpoint = MODEL_CHECKPOINT)
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout= LORA_DROPOUT,
        bias= BIAS,
        modules_to_save=["decode_head"],
        )
    lora_model = depth_anything.peft_model(peft_config)
    checkpoint = torch.load(TRAINED_CHECKPOINT)
    lora_model.load_state_dict(checkpoint['model_state_dict'])

    # Output
    OUTPUT_FILE_BEFORE = "DepthAnything/scripts/eval/without_training.csv"
    model_evaluation(depth_anything.model,IMAGE_PROCESSOR,DATASET_ROOT_DIR,DATA_USE_PERCENTAGE,BATCH_SIZE,OUTPUT_FILE_BEFORE)

    OUTPUT_FILE_AFTER = "DepthAnything/scripts/eval/training_8epochs.csv"
    model_evaluation(lora_model,IMAGE_PROCESSOR,DATASET_ROOT_DIR,DATA_USE_PERCENTAGE,BATCH_SIZE,OUTPUT_FILE_AFTER)

main()

    

