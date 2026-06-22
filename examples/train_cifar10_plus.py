# examples/train_cifar10_plus.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys, os

# Permet d'importer le module hcnn situé dans le dossier parent
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hcnn import HCNNPlus, get_default_config

def main():
    # Chargement de la configuration
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instanciation du modèle Haute Capacité (HCNNPlus)
    model = HCNNPlus(config).to(device)
    
    # Affichage du nombre de paramètres pour vérification
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle HCNNPlus initialisé sur {device}.")
    print(f"Nombre de paramètres entraînables : {nb_params / 1e6:.3f}M")

    # Pipeline d'augmentation des données pour CIFAR-10 (identique au papier original)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255),
                             (63.0/255, 62.1/255, 66.7/255))
    ])
    
    # Pipeline de test (sans augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255),
                             (63.0/255, 62.1/255, 66.7/255))
    ])

    # Chargement des datasets
    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
        batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transform_test),
        batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimiseur SGD avec Momentum et Weight Decay
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Planificateur de taux d'apprentissage (réduction par paliers)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma)
    
    # Fonction de perte (calcule implicitement le LogSoftmax, d'où le retrait du Softmax dans model.py)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"Début de l'entraînement pour {config.epochs} époques...")
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Boucle d'entraînement
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            
        # Mise à jour du taux d'apprentissage
        scheduler.step()

        # Évaluation toutes les 10 époques (ou à la première époque)
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    # Les prédictions utilisent argmax sur la dimension des classes
                    pred = model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    
            acc = 100. * correct / len(test_loader.dataset)
            epoch_loss = running_loss / len(train_loader.dataset)
            
            print(f"Époque {epoch:3d}/{config.epochs} | Perte (Train) : {epoch_loss:.4f} | Précision (Test) : {acc:.2f}%")
            
            # Sauvegarde du meilleur modèle
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'hcnn_plus_best.pth')
                
    print(f"Entraînement terminé. Meilleure précision atteinte : {best_acc:.2f}%. Poids sauvegardés dans 'hcnn_plus_best.pth'.")

if __name__ == '__main__':
    main()