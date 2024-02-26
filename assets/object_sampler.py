import xml.etree.ElementTree as ET

def generate_object_xml(num_collision, object_idx, save_path):
    # Create the root element
    root = ET.Element("mujoco", model="object")

    # Create the 'asset' element
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="object", file="objects/%d/object.obj" % object_idx)

    for i in range(num_collision):
        ET.SubElement(asset, "mesh", name=f"object{i:03d}", file=f"objects/{object_idx}/object{i:03d}.obj")

    # Create the 'worldbody' element
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name="object")

    # Add 'freejoint' and 'geom' elements to 'body'
    ET.SubElement(body, "freejoint", name="object_root")
    object_v = ET.SubElement(body, "geom", mesh="object", type="mesh")
    object_v.set("class", "visual")

    for i in range(num_collision):
        object_c = ET.SubElement(body, "geom", mesh=f"object{i:03d}", type="mesh")
        object_c.set("class", "collision")

    # Create an ElementTree object and write to file
    tree = ET.ElementTree(root)
    tree.write(save_path)