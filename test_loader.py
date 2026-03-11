from preprocessing.preprocess import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders("data/chest_xray")

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))