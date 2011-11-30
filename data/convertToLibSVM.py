"""
Program converts from RCV1 format to libsvm format
"""
import sys, random

if (len(sys.argv) == 6):
	(labels, data, klass, output, outputsize) = sys.argv[1:6]
	outputsize = int(outputsize)
else:
	print "Usage convertToLibSVM [label file] [test/train files comma seperated] [target class] [output file] [outputsize]"
	print "Output size is length of the file outputed." 
	print "The positive and negative examples are equalled. The size is limited by double the number of positive examples."
	sys.exit(0)

print "looking for: ", klass
outputfile = open(output, 'w')

labeldict = {} # dictionary of labels in format id->(1 if in class "klass" otherwise)
labelsfile = open(labels, 'r')

for line in labelsfile:
	(lclass, lid, one) = line.split()
	if(lclass == klass): # If we found the right classification label it
		labeldict[lid] = 1; # You are going to wanna check if the key exists for a negative classification

positives = []
negatives = []

for datan in data.split(","):
	print "Scanning file", datan
	
	datafile = open(datan, 'r')
	for line in datafile: # This is a loop over thousands of lines. Try not to do anything too complicated in here :()
		rid = ""
		i=0
		while line[i]!=' ':
			rid += line[i]
			i=i+1
		if rid in labeldict:
			positives.append("1 " + line[i:].strip() + "\n")
		else:
			negatives.append("0 " + line[i:].strip() + "\n")

if(outputsize > len(positives)*2):
	outputsize = len(positives)*2

for i in xrange(0,outputsize/2):
	outputfile.write(positives[i])

count = 0	
while count < outputsize/2:
	index = random.randint(1,len(negatives)-1)
	outputfile.write(negatives[index])
	count=count+1
	
print "until: ", outputsize/2