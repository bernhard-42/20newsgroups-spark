import json
import sys

with open(sys.argv[1], "r") as fin:
    note = json.load(fin)

with open("Code.md", "w") as fout:
	def fprint(line=""):
		fout.write(line + "\n")

	for paragraph in note["paragraphs"]:
		if paragraph.has_key("text"):
			text = paragraph["text"].split("\n")
			if len(text[0]) > 0 and text[0][0] == "%":
				typ = text[0][1:]
				text = text[1:]
			else: 
				typ = "scala"
			while (len(text[0]) == 0):
				text = text[1:]

			fprint()
			if typ == "md":
				for line in text:
					fprint(line)
			else:
				indent = ""
				if typ == "scala":
					fprint("```scala")
				elif typ == "pyspark":
					fprint("```python")
				elif typ == "sh":
					fprint("```bash")
				for line in text:
					fprint(indent + line)
				fprint("```")
			fprint()