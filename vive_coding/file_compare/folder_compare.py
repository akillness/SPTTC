import os
import xml.etree.ElementTree as ET
from pathlib import Path

def compare_xml_files(original_file, target_file, result_file):
    """Compare two XML files and save the result."""
    try:
        # Parse XML files
        original_tree = ET.parse(original_file)
        target_tree = ET.parse(target_file)
        
        original_root = original_tree.getroot()
        target_root = target_tree.getroot()
        
        # Create dictionaries with text id as key
        original_dict = {elem.get('id'): elem for elem in original_root.findall('.//text')}
        target_dict = {elem.get('id'): elem for elem in target_root.findall('.//text')}
        
        # Create result XML by copying original XML structure
        result_tree = ET.ElementTree(original_root)
        
        # Update text content where there are differences
        for text_id, original_elem in original_dict.items():
            if text_id in target_dict:
                target_elem = target_dict[text_id]
                if original_elem.text != target_elem.text:
                    original_elem.text = target_elem.text
        
        # Save the result
        result_tree.write(result_file, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"Error comparing files {original_file} and {target_file}: {str(e)}")
        return False

def find_matching_target_file(original_filename, target_dir):
    """Find a matching target file that contains '(1)' in its name."""
    base_name = original_filename.replace('.xml', '')
    for target_file in target_dir.glob('*.xml'):
        if '(1)' in target_file.name and base_name in target_file.name:
            return target_file
    return None

def process_folders():
    """Process all XML files in original and target folders."""
    # Create result folder if it doesn't exist
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    # Get list of files in original folder
    original_dir = Path('original')
    target_dir = Path('target')
    
    if not original_dir.exists() or not target_dir.exists():
        print("Original or target folder not found!")
        return
    
    # Process each XML file in original folder
    for original_file in original_dir.glob('*.xml'):
        target_file = find_matching_target_file(original_file.name, target_dir)
        
        if target_file:
            print(f"Processing {original_file.name} with {target_file.name}...")
            result_file = result_dir / original_file.name
            
            if compare_xml_files(original_file, target_file, result_file):
                print(f"Successfully compared and saved result to {result_file}")
            else:
                print(f"Failed to compare {original_file.name}")
        else:
            print(f"No matching file found in target folder for {original_file.name}")

if __name__ == '__main__':
    process_folders() 