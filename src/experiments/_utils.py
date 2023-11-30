import xml.etree.ElementTree as ET


def change_sumo_config_status(file_path, tls_ids, status):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for element in root.iter('tlLogic'):
        if element.get('id') in tls_ids:
            element.set("type", status)

    tree.write(file_path)