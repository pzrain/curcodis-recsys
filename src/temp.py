import os
os.environ["METIS_DLL"] = "~/metis/usr/local/lib/libmetis.so"
import metis
metis.part_graph
print("Hello world")