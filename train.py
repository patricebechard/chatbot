# Loading custom modules
import model
import preprocessing
import utils
from utils import USE_CUDA
from model import loadModel
from preprocessing import loadDataset
from preprocessing import MAX_LENGTH
from preprocessing import BOS_TOKEN, EOS_TOKEN, OOV_TOKEN
from preprocessing import BOS_TOKEN_id, EOS_TOKEN_id, OOV_TOKEN_id

# Loading Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.functional as F
from torch.autograd import Variable

import random

# Load other modules
import time

def trainModel(n_iters=100000, teacher_forcing_ratio=0., print_every=1000,
			   plot_every=100, learning_rate=0.01, max_length=MAX_LENGTH):

    training_pairs, vocab_size, word2ix, ix2word = loadDataset()
    encoder, decoder = loadModel(vocab_size)

    print("Training the model ... ")
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # reset every print_every
    plot_loss_total = 0  # reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair['input']
        target_variable = training_pair['target']

        input_variable = Variable(torch.LongTensor(input_variable).view(-1, 1))
        target_variable = Variable(torch.LongTensor(target_variable).view(-1, 1))
        if USE_CUDA:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()

        loss = trainIter(input_variable, target_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     max_length=max_length, teacher_forcing_ratio=teacher_forcing_ratio)
        print_loss_total += loss
        plot_loss_total += loss

        # Keeping track of average loss and printing results on screen
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # Keeping track of average loss and plotting in figure
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)

            if min(plot_losses) == plot_loss_avg:
                #we save this version of the model
                torch.save(encoder.state_dict(), "encoder.ckpt")
                torch.save(decoder.state_dict(), "decoder.ckpt")

            plot_loss_total = 0

    utils.showPlot(plot_losses)


def trainIter(input_variable, target_variable, encoder, decoder,
			  encoder_optimizer, decoder_optimizer, criterion,
			  max_length=MAX_LENGTH, teacher_forcing_ratio=0.):

	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
	if USE_CUDA:
		encoder_outputs = encoder_outputs.cuda()

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0][0]

	decoder_input = Variable(torch.LongTensor([[BOS_TOKEN_id]]))
	if USE_CUDA:
		decoder_input = decoder_input.cuda()

	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing : Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden)
			loss += criterion(decoder_output, target_variable[di])
			decoder_input = target_variable[di] # Teacher forcing

	else:
		# Without teacher forcing : use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]

			decoder_input = Variable(torch.LongTensor([[ni]]))
			if USE_CUDA:
				decoder_input = decoder_input.cuda()

			loss += criterion(decoder_output, target_variable[di])
			if ni == EOS_TOKEN_id:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0] / target_length

if __name__ == "__main__":

    trainModel(print_every=10)