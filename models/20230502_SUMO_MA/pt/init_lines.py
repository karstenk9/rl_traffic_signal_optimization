import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse("original_flows.rou.xml")
flowroot = tree.getroot()

flows=flowroot.findall("flow")

setup = pd.read_excel("lines_begin.xlsx", sheet_name="ptlines")

setup=setup[["name","begin"]]

CYCLE=3600

for flow in flows:
    name=flow.find("param").get("value")
    setupbegin=setup[setup["name"]==name].reset_index()["begin"][0]
    flow.set("begin",str(setupbegin))
    flow.set("end",str(setupbegin+CYCLE+1))
tree.write("ptflows.rou.xml",encoding="UTF-8",xml_declaration=True)

