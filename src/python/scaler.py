import sys
inp = open(sys.argv[1], 'r')
outp = open(sys.argv[2], 'w')
threshold = float(sys.argv[3])
for line in inp:
	score = float(line)
	if(score>threshold):
		score = 1.0
	else:
		score = score/threshold
	outp.write(str(score)+'\n')

