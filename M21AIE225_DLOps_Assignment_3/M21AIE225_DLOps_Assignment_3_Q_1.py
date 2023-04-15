

# %%
import pandas as pd
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from sklearn import metrics
import seaborn as sns
import time
import math
import numpy as np
import matplotlib.pyplot as plt
print('Starting ...')


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Available device ==> {device}')


# %%


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# %%
# Train the model
def model_training(model, criterion, optimizer, trainloader, testloader, num_epochs=10, model_name='model'):
    start = time.time()
    loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += metrics.accuracy_score(labels.cpu().detach(
            ).numpy(), outputs.cpu().detach().numpy().argmax(axis=1))
        # Evaluate the model on the validation set
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += metrics.accuracy_score(labels.cpu().detach(
                ).numpy(), outputs.cpu().detach().numpy().argmax(axis=1))
        train_loss = train_loss/len(trainloader)
        val_loss = val_loss/len(testloader)
        train_acc = train_acc/len(trainloader)
        val_acc = val_acc/len(testloader)
        print(f'Epoch: {epoch+1} ({timeSince(start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, ', flush=True)
        loss_list.append([train_loss, val_loss, train_acc, val_acc])

    print(
        f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
    return np.array(loss_list), time.time()-start, loss_list[-1][2], loss_list[-1][3]


# %%
#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred_probs, classes, title=''):
    Y_pred = Y_pred_probs.argmax(axis=1)
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    f1_score = metrics.f1_score(Y_test, Y_pred, average='weighted')

    plt.figure(figsize=(10, 10))
    plt.title(
        f'{title} : Accuracy : {overallAccuracy*100:3.2f}% | F1 Score : {f1_score*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues',
                fmt='g', xticklabels=classes, yticklabels=classes)

    plt.show()
    plt.savefig(
        f'confusion_mat_{title}_{time.time()}.png', bbox_inches='tight')
    print(f'Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'F1 Score: {f1_score*100:3.3f}%')
    classwiseAccuracy_df = pd.DataFrame(
        data=[classwiseAccuracy], columns=classes)
    print(
        f'\nClasswise Accuracy Score: \n{classwiseAccuracy_df.to_markdown(index=False)}')
    print('\nConfusion Matrix:')
    print(cm.to_markdown())


# %%
def plot_training_graphs(loss_list, title=''):
    fig = plt.figure(figsize=(20, 7))
    plot = fig.add_subplot(1, 2, 1)
    plot.set_title(f"{title} : Training vs Validation loss")
    plot.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
    plot.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Loss")
    plot.legend()
    plot = fig.add_subplot(1, 2, 2)
    plot.set_title(f"{title} : Training vs Validation Accuracy")
    plot.plot(loss_list[:, 2], linestyle='--', label="Training Accuracy")
    plot.plot(loss_list[:, 3], linestyle='-', label="Validation Accuracy")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Accuracy")
    plot.legend()
    plt.show()
    plt.savefig(
        f'training_loss_{title}_{time.time()}.png', bbox_inches='tight')


# %%

def split_patches_from_images(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size,
                              j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MultiHeadSelfAttentionBlock(torch.nn.Module):
    def __init__(self, d, n_heads=2, activation_fn=torch.nn.Softmax):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = torch.nn.ModuleList(
            [torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = torch.nn.ModuleList(
            [torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = torch.nn.ModuleList(
            [torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.activation_fn = activation_fn(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.activation_fn(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class VisionTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim, n_heads, mlp_ratio=4, mlp_activation=torch.nn.GELU, msa_activation=torch.nn.Softmax):
        super(VisionTransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttentionBlock(
            hidden_dim, n_heads, msa_activation)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            mlp_activation(),
            torch.nn.Linear(mlp_ratio * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.pe = torch.nn.Parameter(torch.zeros(max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class VisionTransformerClassifier(torch.nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_dim=8, n_heads=2, out_d=10, activation_fn=torch.nn.ReLU, learn_pos_emb=False):
        # Super constructor
        super(VisionTransformerClassifier, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.mlp_activation = torch.nn.GELU if activation_fn == None else activation_fn
        self.msa_activation = torch.nn.Softmax if activation_fn == None else activation_fn
        self.learn_pos_emb = learn_pos_emb

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = torch.nn.Linear(self.input_d, self.hidden_dim)

        # 2) Learnable classification token
        self.class_token = torch.nn.Parameter(torch.rand(1, self.hidden_dim))

        # 3) Positional embedding
        if learn_pos_emb:
            self.positional_embeddings = PositionalEmbedding(
                hidden_dim, n_patches ** 2 + 1)
        else:
            self.register_buffer('positional_embeddings', get_positional_embeddings(
                n_patches ** 2 + 1, hidden_dim), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = torch.nn.ModuleList(
            [VisionTransformerBlock(hidden_dim, n_heads, mlp_activation=activation_fn) for _ in range(n_blocks)])

        # 5) Classification MLPk
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, out_d),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = split_patches_from_images(images, self.n_patches).to(
            self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        # Map to output dimension, output category distribution
        return self.mlp(out)


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(
                i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
# %%


def view_samples(testloader, classes):

    test_images, labels = next(iter(testloader))
    print(test_images.shape)
    fig = plt.figure(figsize=(20, 7))
    nrows = 3
    ncols = 10
    b = np.random.randint(0, test_images.shape[0]-nrows*ncols)
    for i in range(nrows*ncols):
        plot = fig.add_subplot(nrows, ncols, i+1)
        plot.set_title(classes[labels[i+b].cpu().numpy()])
        plot.imshow(np.transpose(test_images[i+b], (1, 2, 0)).cpu())
    plt.show()
    plt.savefig(f'view_samples_{time.time()}.png', bbox_inches='tight')


def filter_dataset(dataset_full):
    # Selecting even classes 0,2,4,6,8 as roll number is odd (M21AIE225)
    targets = np.array(dataset_full.targets)
    idx = (targets == 0) | (targets == 2) | (
        targets == 4) | (targets == 6) | (targets == 8)
    dataset_full.targets = np.rint(targets[idx]/2).astype(int)
    dataset_full.data = dataset_full.data[idx]
    dataset_full.classes = [dataset_full.classes[c] for c in [0, 2, 4, 6, 8]]
    return dataset_full


# %%
print('Data Loading ...')
transform = T.Compose([T.ToTensor()])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_set = filter_dataset(train_set)
trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_set = filter_dataset(test_set)
testloader = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=True)

view_samples(trainloader, train_set.classes)
print('Data Loading Done !')

# %%

activation_fns = [torch.nn.ReLU, torch.nn.Tanh, torch.nn.GELU]

positional_embeddings = [
    False,  # for cosine positional embedding
    True,  # for learnable positional embedding
]

results = []

for pe in positional_embeddings:
    for act_fn in activation_fns:
        print(
            f'\n\n***\t\t\t\tActivation Function = {act_fn.__name__}\tLearnable Posisional Embedding = {pe}\t\t\t\t***\n\n')

        n_patches = 8
        n_blocks = 6
        hidden_dim = 8
        n_heads = 8

        if pe:
            n_blocks = 4
            n_heads = 6
            hidden_dim = 12
        else:
            n_blocks = 6
            n_heads = 8
            hidden_dim = 8

        model = VisionTransformerClassifier((3, 32, 32), n_patches=n_patches, n_blocks=n_blocks,
                                            hidden_dim=hidden_dim, n_heads=n_heads, out_d=10, activation_fn=act_fn).to(device)

        print(
            f'Model created with parameter  n_blocks = { n_blocks }, n_heads = { n_heads }, hidden_dim = { hidden_dim }, act_fn = { act_fn.__name__ }, positional embd = { "Learnable" if pe else "Cosine" }, ')
        # print(model)
        model_name = f'ViT_{act_fn.__name__}_{ "Learnable" if pe else "Cosine" }_{n_blocks}B{n_heads}H{hidden_dim}D'
        print(f'Starting training {model_name}!')
        lr = 1e-4
        epoch = 10
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f'optimizer => {optimizer}')
        criterion = torch.nn.CrossEntropyLoss()
        print(f'criterion => {criterion}')
        print(f'learning rate => {lr}')
        print(f'epoch => {epoch}')
        loss_list, t, train_a, test_a = model_training(
            model, criterion, optimizer, trainloader, testloader, num_epochs=epoch, model_name='VisionTransformer')
        plot_training_graphs(loss_list, title=model_name)
        model.eval()
        with torch.no_grad():
            test_labels = []
            test_output = []
            for batch in testloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                test_labels += y.cpu()
                test_output += torch.argmax(y_hat, dim=1).cpu()

            test_labels = np.array(test_labels)
            test_output = np.array(test_output)
            print(f'\nModel Evaluation Summary:')
            confusionMatrixAndAccuracyReport(
                test_labels, test_output, test_set.classes, title=model_name)
        results.append({
            "model": model_name,
            "Positional Embed": 'Learnable PE' if pe else 'Cosine PE',
            "Activation Fn": act_fn.__name__,
            "Learning Rate": lr,
            "Epoch": epoch,
            "Training Time": t,
            "Training Accuracy": train_a,
            "Test Accuracy": test_a,
        })

        df = pd.DataFrame(results)
        df.to_csv("M21AIE225_Ass_3_Q1_resuts_1.csv", index=False)


# %%

print("\n\n Overall Summary : \n")
print(df.to_markdown(index=False))
