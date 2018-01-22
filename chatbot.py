from sys import argv

print(argv)

if len(argv) > 2:
	raise Exception("A maximum of 1 argument can be called at a time. " + \
					"You have called %d arguments : %s" % (len(argv) - 1, str(argv[1:])))

if "-on" in argv or "--online" in argv:
	import online
	online.online()

elif "-tr" in argv or "--train" in argv:
	import train
	train.trainModel()

elif "-te" in argv or "--test" in argv:
	import test
	test.testModel()