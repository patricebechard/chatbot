from preprocessing import MAX_LENGTH

def online(encoder, decoder, word2ix, ix2word, max_length=20):

	while True:
		
		query = input("-> ")
		query = preprocessing.normalizeString(query)
		query = preprocessing.convertWordSentenceToIndex(query, 
														 word2ix=word2ix)
		query.append(EOS_TOKEN_ix)

		query = Variable(torch.LongTensor(query).view(-1, 1))
		if USE_CUDA:
			query = query.cuda()

		if decoder.attn_status:
			output_words, attentions = evaluate(encoder, decoder, query,
												word2ix, ix2word,
		                                    	max_length=max_length)
		else:
			output_words = evaluate(encoder, decoder, query,
									word2ix, ix2word, max_length=max_length)
		output_words = output_words[:-1] #strip the <EOS> token
		output_sentence = ' '.join(output_words)

		print(output_sentence)
