

def decorator_training_results(func):
    def wrapper(*args, **kwargs):
        train_results = func(*args, **kwargs)
        epoch, train_loss, train_acc = train_results
        print(f"[ Epoch {epoch} ] train_loss: {train_loss:.4}, train_acc: {train_acc:.4}")
        return train_results
    return wrapper


def decorator_classification_report(func):
    def wrapper(*args, **kwargs):
        validation_results = func(*args, **kwargs)
        val_loss, val_acc, f1, f1_macro, report, report_p, cm = validation_results
        print(f"validation: val_loss: {val_loss:.4}, val_acc: {val_acc:.4}")
        print("classification_report:")
        print(report_p)
        return validation_results
    return wrapper

