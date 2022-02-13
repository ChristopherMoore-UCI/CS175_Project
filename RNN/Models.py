import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU(nn.Module):
		def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
			super(GRU, self).__init__()
			self.num_layers = num_layers
			self.hidden_size = hidden_size
			self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
			self.out_layer = nn.Linear(hidden_size, num_classes)

		def forward(self, x):
			hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
			out, _ = self.gru(x, hidden_0)

			#Format out dims
			return self.out_layer(out[:, -1, :])

class SumRNN(nn.Module):
		def __init__(self, input_size, hidden_size, num_layers=2, num_classes=2):
			super(SumRNN, self).__init__()
			#Add embedding Layer
			self.num_layers = num_layers
			self.hidden_size = hidden_size
			self.gru_word_level = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) #Why does this need to be batch first
			self.gru_sentence_level = nn.GRU(hidden_size, hidden_size, num_layers) #Make bidirectional, and maybe batch first?
			self.pool = nn.AvgPool2d((2,1), stride=(2,1))
			self.out_layer = nn.Linear(hidden_size, num_classes)

		def forward(self, x):
			hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
			
			gru_forward = self.gru_word_level(x, hidden_0)
			gru_backward = self.gru_word_level(torch.flip(x,[0]), hidden_0)
			cat = torch.cat((gru_forward[1], gru_backward[1]),1)
			word_level_out = self.pool(cat)
			sentence_level_out = self.gru_sentence_level(word_level_out, hidden_0)
			

			#Format out dims
			return self.out_layer(sentence_level_out[0].permute(1, 0, 2)[:, -1, :]) #Remove permute for final version

