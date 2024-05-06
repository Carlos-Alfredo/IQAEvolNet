import torch
import torch.nn as nn
import image_data_loader
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pretrain(model,training_dataset,validation_dataset,epoch,learning_rate,folder):

    model.apply(weights_init)

    training_size = training_dataset.shape[0]
    validation_size = validation_dataset.shape[0]

    training_data = image_data_loader.hazy_data_loader(training_dataset,validation_dataset)
    validation_data = image_data_loader.hazy_data_loader(training_dataset,validation_dataset, mode="val")

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0001)

    model.train()

    num_of_epochs = int(epoch)
    for epoch in range(num_of_epochs):
        for iteration, data in enumerate(training_data_loader):
            hazy_image = (data[:,0].unsqueeze(0)).permute(1,0,2,3)
            hazefree_image = (data[:,1].unsqueeze(0)).permute(1,0,2,3)
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            dehaze_image = model(hazy_image)

            loss = criterion(dehaze_image, hazefree_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
            optimizer.step()

            if ((iteration+1) % 10) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % 200) == 0:
                torch.save(model.state_dict(), folder + "Epoch" + str(epoch) + '.pth')
            else:
                torch.save(model.state_dict(), folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        for iter_val, data in enumerate(validation_data_loader):
            hazy_image = (data[:,0].unsqueeze(0)).permute(1,0,2,3)
            hazefree_image = (data[:,1].unsqueeze(0)).permute(1,0,2,3)

            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            dehaze_image = model(hazy_image)

            # torchvision.utils.save_image(torch.cat((hazy_image, dehaze_image, hazefree_image),0), "training_data_captures/" +str(iter_val+1)+".jpg")

        torch.save(model.state_dict(), folder + "trained_LDNet.pth")
    return model.state_dict()