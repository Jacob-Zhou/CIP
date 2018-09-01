from NN import NN
from config import config

iterations = config["iterations"]
break_num=config["breakNum"]
shuffle=config["shuffle"]
step_opt=config["step opt"]
batchsize=config["batchsize"]
max_entropy=config["Max Entropy"]
learnrate=config["learnrate"]
regularization=config["regularization"]

nn = NN()
nn.read_data()
nn.create_feature_space()
nn.train(iteration=iterations,break_num=break_num,shuffle=shuffle,step_opt=step_opt,batch_size=batchsize,regularization=regularization,learnrate=learnrate,max_entropy=max_entropy)
