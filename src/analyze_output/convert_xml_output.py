import pandas as pd
import lxml.etree as ET
import xml.etree.ElementTree as ET
import os


def parse_XML(xml_file, df_cols): 
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    """
    out_df = pd.DataFrame(columns = df_cols)
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()

    #i = 0
    
    #while i <= len(xroot):
    for node in xroot: 
        rows = []
        j = 0
        for element in node:
            res = []
            res.append(node.attrib.get(df_cols[0]))
            
            for el in df_cols[1:]:
                if node is not None and element.get(el) is not None:
                    res.append(element.get(el))
                else: 
                    res.append(None)
            rows.append({df_cols[i]: res[i] 
                        for i, _ in enumerate(df_cols)})
    
            out_df.loc[len(out_df)] = rows[j] 
            j += 1
    
    out_df.to_csv(f'urban_mobility_simulation/src/data/actuated_output/{xml_file}.csv', index=False)
    print('Output saved to: ', f'{xml_file}.csv')


if __name__ == '__main__':
    df_cols = ["time",
               "id",
               "eclass",
               "CO2",
               "CO",
               "HC",
               "PMx",
               "NOx",
               "PMx",
               "fuel",
               "electricity",
               "route",
               "type",
               "noise",
               "waiting",
               "lane",
               "pos",
               "speed",
               "angle",
               "x",
               "route",
               "y"] 
    
    inputdir = '/pfs/data5/home/ma/ma_ma/ma_jenhahn/masterarbeit_sumo/urban_mobility_simulation/src/data/actuated_output/'
    for file in os.listdir(inputdir):
        if file.endswith('emission_info_actuated_TL.xml'):
            print(file)
            parse_XML(inputdir + file, df_cols)