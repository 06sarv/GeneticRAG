import sys, getopt

def buildAlleleDict(file):
	returnDict = {}
	for line in file:
		l = line.split('\t')
		print(l)
		print(len(l))
		if len(l) == 8:
			returnDict[':'.join(l[1:6])] = l[7].strip() 
			print("returnDict key: " + ':'.join(l[1:6]))
	return returnDict

def annotate(filein, fileout, alleleDict):
	with open(filein, 'r') as input_vcf, open(fileout, 'w') as output_vcf:
		header_lines = []
		variant_lines = []
		for line in input_vcf:
			if line.startswith('##'):
				header_lines.append(line)
			elif line.startswith('#CHROM'):
				header_lines.append('##INFO=<ID=indiaFreq,Number=1,Type=String,Description="Allele Frequency from India Allele Finder">\n')
				header_lines.append(line)
			else:
				variant_lines.append(line)

		for h_line in header_lines:
			output_vcf.write(h_line)

		for line in variant_lines:
			l = line.strip().split('\t')
			id = ':'.join(l[0:5])
			info_field = l[7]
			if id in alleleDict:
				print("match: " + alleleDict[id])
				info_field += f";indiaFreq={alleleDict[id].strip()}"
			else:
				print("no match for id: " + id + "... printing N/A")
				info_field += ";indiaFreq=N/A"
			l[7] = info_field
			output_vcf.write('\t'.join(l) + '\n')

def main(argv):
	global alleleDict
	input = '' #points to a vcf file
	output = '' #output vcf file
	alleles_file = '' #allele frequency file

	try:
		opts, args = getopt.getopt(argv,'hi:o:f:', ['input=', 'output=', 'alleles='])
	except getopt.GetoptError:
		print('indiaAlleleAnnotator.py -i <infile> -o <outfile> -f <alleles_file>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('indiaAlleleAnnotator.py -i <infile> -o <outfile> -f <alleles_file>')
			sys.exit()
		elif opt in ('-i', '--input'):
			input = arg
		elif opt in ('-o', '--output'):
			output = arg
		elif opt in ('-f', '--alleles'):
			alleles_file = arg
	
	alleleDict = buildAlleleDict(open(alleles_file, 'r'))

	annotate(input, output, alleleDict)

if __name__ == "__main__":
	main(sys.argv[1:])
