import argparse
import importlib

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a specified model from the model folder.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["CNN", "CNN_LSTM", "LSTM", "Transformer"],
        help="Select the model to train: CNN, CNN_LSTM, LSTM, or Transformer"
    )
    args = parser.parse_args()

    # Construct module name assuming the model files are in the 'model' folder
    module_name = f"model.{args.model}"
    print(f"Importing module: {module_name}")

    try:
        # Dynamically import the selected model module
        model_module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        return

    # Look for a training entry point in the module:
    # First try train_model(), then try main()
    if hasattr(model_module, "train_model"):
        print(f"Starting training using {args.model}.train_model()...")
        model_module.train_model()
    elif hasattr(model_module, "main"):
        print(f"Starting training using {args.model}.main()...")
        model_module.main()
    else:
        print(f"Module {module_name} does not have a recognized training function (train_model() or main()).")

if __name__ == "__main__":
    main()
