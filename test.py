def testIter(encoder, decoder, sentence, word2ix, ix2word, max_length=20):
	input_variable = sentence
	input_length = input_variable.size()[0]
	encoder_hidden = encoder.initHidden()

	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
	if use_cuda:
		encoder_outputs = encoder_outputs.cuda()

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_variable[ei],
		                                         encoder_hidden)
		encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

	decoder_input = Variable(torch.LongTensor([[BOS_TOKEN_ix]]))
	if use_cuda:
		decoder_input = decoder_input.cuda()

	decoder_hidden = encoder_hidden

	decoded_words = []
	if decoder.attn_status:
		decoder_attentions = torch.zeros(max_length, max_length)

	for di in range(max_length):
		if decoder.attn_status:
			decoder_output, decoder_hidden, decoder_attention = decoder(
			  	  decoder_input, decoder_hidden, encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
		else:
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		if ni == EOS_TOKEN_ix:
			decoded_words.append(EOS_TOKEN)
			break
		else:
			decoded_words.append(ix2word[str(ni)])

		decoder_input = Variable(torch.LongTensor([[ni]]))
		if use_cuda:
			decoder_input = decoder_input.cuda()

	if decoder.attn_status:
		return decoded_words, decoder_attentions[:di + 1]
	else:
		return decoded_words

def testModel(encoder, decoder, dataset, word2ix, ix2word, 
					 n=10, max_length=20):
    for i in range(n):
        pair = random.choice(dataset)
        input_str = ''.join(e for e in \
        	preprocessing.convertIndexSentenceToWord(pair['input']))
        target_str = ''.join(e for e in \
        	preprocessing.convertIndexSentenceToWord(pair['target']))
        print('>', input_str)
        print('=', target_str)

        temp = Variable(torch.LongTensor(pair['input']).view(-1, 1))
        if USE_CUDA:
        	temp = temp.cuda()

        output_words, attentions = evaluate(encoder, decoder, temp,
        									word2ix, ix2word,
                                            max_length=max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

encoder.load_state_dict(torch.load("encoder.ckpt"))
decoder.load_state_dict(torch.load("decoder.ckpt"))