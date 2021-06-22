# -*- coding: utf-8 -*-
import pickle5

import imageio
import matplotlib
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from loader import loader
from models.model_factory import model_factory
from transformations.transform_factory import transform_factory

matplotlib.style.use('ggplot')


class test_architectur():
    def __init__(self,
                 ds_name="fmnist",
                 lr_generator=0.0002,
                 lr_discriminator=0.0002,
                 batch_size=32,
                 epochs=100,
                 sample_size=64,
                 nz=16,
                 k=1,
                 embedding_name="distilbert",  # "fasttext" or "distilbert"
                 text_to_image=False,
                 model_save_interval=50,
                 mlflow_tags=[]
                 ):

        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.sample_size = sample_size  # fixed sample size
        self.nz = nz  # latent vector size
        self.k = k  # number of steps to apply to the discriminator
        self.model_save_interval = model_save_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlflow_tags = mlflow_tags
        self.embedding_name = embedding_name
        self.textToImage = text_to_image

    def train(self):
        # specify dataset name

        # parameter for our approach "Text to Images"
        embed_dict = None
        embed_dim = None
        embeddings = None
        embeddings_type = None

        if self.textToImage:
            if self.ds_name != 'coco':
                # load dict for mnist and cifar embeddings
                with open('./embeddings/fasttext/' + self.ds_name + '_embeddings.pickle', 'rb') as fin:
                    embed_dict = pickle5.load(fin)
                embed_dim = embed_dict[0].size
            else:
                if self.embedding_name == "distilbert":
                    embeddings_type = 'distilbert-base-uncased'
                    embeddings_dir = 'transformers/'
                else:
                    embeddings_type = 'fasttext'
                    embeddings_dir = 'fasttext/'

        mlflow.set_experiment(self.ds_name)
        mlflow.end_run()
        mlflow.start_run()
        mlflow_experiment_id = mlflow.get_experiment_by_name(self.ds_name).experiment_id
        mlflow_run_id = mlflow.active_run().info.run_id
        log_path = "mlruns/" + str(mlflow_experiment_id) + "/" + str(mlflow_run_id) + "/" + "artifacts" + "/"
        mlflow.log_param("run_id", mlflow_run_id)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("sample_size", self.sample_size)
        mlflow.log_param("nz", self.nz)
        mlflow.log_param("k", self.k)
        mlflow.log_param("textToImage", self.textToImage)
        mlflow.log_param("device", self.device)
        mlflow.log_param("model_save_interval", self.model_save_interval)
        #mlflow.log_param("text_to_image", self.text_to_image)
        mlflow.log_param("embedding", self.embedding_name)

        if len(self.mlflow_tags) != 0:
            mlflow.set_tags(self.mlflow_tags)

        print("mlflow logpath:" + log_path)

        transform = transform_factory.transform_factory(self.ds_name).get_compose()
        to_pil_image = transforms.ToPILImage()

        if self.ds_name != 'coco':
            train_data = loader(self.ds_name, transform).getDataset()
        else:
            # use cifar10 classes excepting deer and frog
            categories = ['cat', 'dog', 'bird', 'horse', 'airplane', 'truck', 'boat', 'car']
            train_data = loader(self.ds_name, transform, self.textToImage, categories, sample_size,
                                embeddings_type).getDataset()

            if self.textToImage:
                embed_dim = train_data.getEmbeddingDim()
                with open('./embeddings/' + embeddings_dir + 'coco_' + embeddings_type + '_embeddings.pickle',
                          'rb') as fin:
                    embed_dict = pickle5.load(fin)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                  drop_last=True)  # set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
        generator = model_factory.generator_factory(self.ds_name, nz, self.textToImage, embed_dim).to(self.device)
        discriminator = model_factory.discriminator_factory(self.ds_name, self.textToImage, embed_dim).to(self.device)

        print('##### GENERATOR #####')
        print(generator)
        print('######################')

        print('\n##### DISCRIMINATOR #####')
        print(discriminator)
        print('######################')

        # optimizers
        optim_g = optim.Adam(generator.parameters(), lr=0.0002)
        optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

        # loss function
        criterion = nn.BCELoss()  # binary cross entropy

        losses_g = []  # to store generator loss after each epoch
        losses_d = []  # to store discriminator loss after each epoch
        images = []  # to store images generatd by the generator

        # to create real labels (1s)
        def label_real(size):
            data = torch.ones(size, 1)
            return data.to(self.device)

        # to create fake labels (0s)
        def label_fake(size):
            data = torch.zeros(size, 1)
            return data.to(self.device)

        # function to create the noise vector
        def create_noise(sample_size, nz):
            return torch.randn(sample_size, nz).to(self.device)

        # to save the images generated by the generator
        def save_generator_image(image, path):
            save_image(image, path)

        # function to train the discriminator network
        def train_discriminator(optimizer, data_real, data_fake, embedding):
            b_size = data_real.size(0)
            real_label = label_real(b_size)
            fake_label = label_fake(b_size)

            optimizer.zero_grad()

            output_real = discriminator(data_real, embedding)
            loss_real = criterion(output_real, real_label)

            output_fake = discriminator(data_fake, embedding)
            loss_fake = criterion(output_fake, fake_label)

            loss_real.backward()
            loss_fake.backward()
            optimizer.step()

            return loss_real + loss_fake

        # function to train the generator network
        def train_generator(optimizer, data_fake, embedding):
            b_size = data_fake.size(0)
            real_label = label_real(b_size)

            optimizer.zero_grad()

            output = discriminator(data_fake, embedding)
            loss = criterion(output, real_label)

            loss.backward()
            optimizer.step()

            return loss

        # map labels to the corresponding labels of CIFAR10, MNIST or FMNIST
        # TODO expand for COCO sentences
        def get_embeddings(labels, idx=None):
            if self.ds_name != 'coco':
                embeddings = [embed_dict[int(label.detach())] for label in labels]
            else:
                embeddings = []
                i = idx.detach().numpy()
                l = labels.detach().numpy()
                for j in range(len(i)):
                    embeddings.append(embed_dict[i[j]][l[j]])
            embeddings = np.array(embeddings)
            return torch.from_numpy(embeddings)

        # generates random labels from range(10).
        # Used for MNIST, FMNIST and CIRFAR
        def create_random_labels(sample_size):
            if self.ds_name != 'coco':
                random_labels = np.random.choice(10, (sample_size))
                random_embed = [embed_dict[int(label)] for label in np.nditer(random_labels)]
                random_embed = np.array(random_embed).reshape(sample_size, -1)
                return torch.from_numpy(random_embed).to(self.device)
            else:
                embeddings = []
                for data in train_data:
                    _, idx, labels = data
                    embeddings.append(embed_dict[idx][labels])
                embeddings = np.array(embeddings)
                return torch.from_numpy(embeddings).to(self.device)

        # create the noise vector and labels
        noise = create_noise(sample_size, nz)
        if self.textToImage:
            # generate random labels for testing
            embeddings = create_random_labels(sample_size)

        # generator.train()
        # discriminator.train()

        for epoch in range(self.epochs):
            loss_g = 0.0
            loss_d = 0.0
            embed_batch = None
            for bi, data in enumerate(train_loader):
                if self.textToImage:
                    if self.ds_name != 'coco':
                        image, labels = data
                        embed_batch = get_embeddings(labels).to(self.device)
                    else:
                        image, idx, labels = data
                        embed_batch = get_embeddings(labels, idx).to(self.device)
                else:
                    image, _ = data

                image = image.to(self.device)
                b_size = len(image)

                #         run the discriminator for k number of steps
                for step in range(self.k):
                    data_fake = generator(create_noise(b_size, nz), embed_batch).detach()
                    data_real = image
                    # train the discriminator network
                    loss_d += train_discriminator(optim_d, data_real, data_fake, embed_batch)

                data_fake = generator(create_noise(b_size, nz), embed_batch)
                # train the generator network
                loss_g += train_generator(optim_g, data_fake, embed_batch)

            # create the final fake image for the epoch
            generated_img = generator(noise, embeddings).cpu().detach()
            # make the images as grid
            generated_img = make_grid(generated_img)

            # save the generated torch tensor models to disk
            save_generator_image(generated_img, log_path + "gen_img" + str(epoch) + ".png")
            images.append(generated_img)
            epoch_loss_g = loss_g / bi  # total generator loss for the epoch
            epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
            losses_g.append(epoch_loss_g.cpu().detach().numpy())
            losses_d.append(epoch_loss_d.cpu().detach().numpy())

            mlflow.log_metric("loss_generator", losses_g[-1].item(), epoch)
            mlflow.log_metric("loss_discriminator", losses_d[-1].item(), epoch)
            if epoch % self.model_save_interval == 0:  # each model is 60mb in size
                torch.save(generator.state_dict(), log_path + "generator" + str(epoch) + ".pth")
                torch.save(discriminator.state_dict(), log_path + "discriminator" + str(epoch) + ".pth")

            print(f"Epoch {epoch} of {epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss {epoch_loss_d:.8f}")

        print('DONE TRAINING')
        torch.save(generator.state_dict(), log_path + "generator" + str(self.epochs) + ".pth")
        torch.save(discriminator.state_dict(), log_path + "discriminator" + str(self.epochs) + ".pth")

        # save the generated images as GIF file
        imgs = [np.array(to_pil_image(img)) for img in images]
        # imageio.mimsave('outputs/generator_images.gif', imgs)
        imageio.mimsave(log_path + 'generator_images.gif', imgs)
        # plot and save the generator and discriminator loss
        plt.figure()
        plt.plot(losses_g, label='Generator loss')
        plt.plot(losses_d, label='Discriminator Loss')
        plt.legend()
        # plt.savefig('outputs/loss.png')
        plt.savefig(log_path + 'loss.png')

if __name__ == "__main__":

    mlflow_tags = {"benchmark": "21_06_2021"}

    for text_to_image in [True]:
        for embedding_name in ["distilbert", "fasttext"]:
            for sample_size in [32, 64, 128, 256]:
                for nz in [32, 64, 128, 256]:
                    setup = test_architectur(
                        sample_size=sample_size,
                        nz=nz,
                        embedding_name=embedding_name,
                        text_to_image=text_to_image,
                        mlflow_tags=mlflow_tags
                    )
                    setup.train()
