import io
p="main.py"
s=io.open(p,encoding="utf-8").read()
s=s.replace(chr(34)+"qwen3:8b"+chr(34), chr(34)+"qwen3-cpu"+chr(34))
s=s.replace(chr(34)+"qwen3-vl:8b"+chr(34), chr(34)+"qwen3-vl-cpu"+chr(34))
s=s.replace(chr(34)+"bge-m3"+chr(34), chr(34)+"bge-m3-cpu"+chr(34))
io.open(p,"w",encoding="utf-8").write(s)
print("done")
