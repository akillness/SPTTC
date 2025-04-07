import sys
import difflib
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTextEdit, QFileDialog, 
                           QLabel, QScrollArea, QFrame, QTabWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat, QFont, QTextBlockFormat

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        self.editor.lineNumberAreaPaintEvent(event)

class FileCompareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('파일 비교 도구')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        self.original_btn = QPushButton('원본 파일 선택')
        self.compare_btn = QPushButton('비교 파일 선택')
        self.save_btn = QPushButton('결과 저장')
        self.compare_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.original_btn)
        button_layout.addWidget(self.compare_btn)
        button_layout.addWidget(self.save_btn)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create comparison tab
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        
        # Create text areas layout for comparison
        text_layout = QHBoxLayout()
        
        # Create text areas with line numbers
        self.original_text = self.create_text_editor()
        self.compare_text = self.create_text_editor()
        
        # Set read-only
        self.original_text.setReadOnly(True)
        self.compare_text.setReadOnly(True)
        
        # Add line numbers
        self.original_text.setLineWrapMode(QTextEdit.NoWrap)
        self.compare_text.setLineWrapMode(QTextEdit.NoWrap)
        
        # Connect scroll signals
        self.original_text.verticalScrollBar().valueChanged.connect(self.sync_scroll)
        self.compare_text.verticalScrollBar().valueChanged.connect(self.sync_scroll)
        
        text_layout.addWidget(self.original_text)
        text_layout.addWidget(self.compare_text)
        
        comparison_layout.addLayout(text_layout)
        
        # Create result tab
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        
        # Create result text area
        self.result_text = self.create_text_editor()
        self.result_text.setReadOnly(True)
        self.result_text.setLineWrapMode(QTextEdit.NoWrap)
        
        result_layout.addWidget(self.result_text)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(comparison_tab, "파일 비교")
        self.tab_widget.addTab(result_tab, "결과")
        
        # Add layouts to main layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tab_widget)
        
        # Connect buttons to functions
        self.original_btn.clicked.connect(self.load_original_file)
        self.compare_btn.clicked.connect(self.load_compare_file)
        self.save_btn.clicked.connect(self.save_result)
        
        self.original_file = None
        self.compare_file = None
        
    def create_text_editor(self):
        editor = QTextEdit()
        editor.setFont(QFont('맑은 고딕', 10))
        editor.setFrameStyle(QFrame.NoFrame)
        return editor
        
    def load_original_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '원본 파일 선택', '', 'XML Files (*.xml)')
        if file_name:
            self.original_file = file_name
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            self.original_text.setText(content)
            self.compare_btn.setEnabled(True)
            
    def load_compare_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '비교 파일 선택', '', 'XML Files (*.xml)')
        if file_name:
            self.compare_file = file_name
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            self.compare_text.setText(content)
            self.compare_files()
            self.save_btn.setEnabled(True)
            
    def compare_files(self):
        if not self.original_file or not self.compare_file:
            return
            
        # Parse XML files
        original_tree = ET.parse(self.original_file)
        compare_tree = ET.parse(self.compare_file)
        
        original_root = original_tree.getroot()
        compare_root = compare_tree.getroot()
        
        # Create dictionaries with text id as key
        original_dict = {elem.get('id'): elem for elem in original_root.findall('.//text')}
        compare_dict = {elem.get('id'): elem for elem in compare_root.findall('.//text')}
        
        # Clear previous formatting
        self.original_text.clear()
        self.compare_text.clear()
        self.result_text.clear()
        
        # Display original and compare files with highlighting
        line_number = 1
        for elem in original_root.findall('.//text'):
            text_id = elem.get('id')
            text_content = elem.text if elem.text else ''
            
            # Original file display
            self.original_text.append(f"{line_number:4d} {text_id}: {text_content}")
            
            # Compare file display
            if text_id in compare_dict:
                compare_elem = compare_dict[text_id]
                compare_content = compare_elem.text if compare_elem.text else ''
                self.compare_text.append(f"{line_number:4d} {text_id}: {compare_content}")
                
                # Highlight differences
                if text_content != compare_content:
                    cursor = self.original_text.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    format = QTextCharFormat()
                    format.setBackground(QColor(255, 200, 200))
                    cursor.mergeCharFormat(format)
                    
                    cursor = self.compare_text.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    format = QTextCharFormat()
                    format.setBackground(QColor(200, 255, 200))
                    cursor.mergeCharFormat(format)
            else:
                self.compare_text.append(f"{line_number:4d} ")
                cursor = self.original_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                format = QTextCharFormat()
                format.setBackground(QColor(255, 200, 200))
                cursor.mergeCharFormat(format)
            
            line_number += 1
        
        # Create result XML by copying original XML structure
        result_tree = ET.ElementTree(original_root)
        
        # Update only the text content where there are differences
        # Keep original content for text ids that don't exist in compare file
        for text_id, original_elem in original_dict.items():
            if text_id in compare_dict:
                compare_elem = compare_dict[text_id]
                if original_elem.text != compare_elem.text:
                    original_elem.text = compare_elem.text
            # If text id doesn't exist in compare file, keep original content
        
        # Convert result to string with proper formatting
        result_str = ET.tostring(result_tree.getroot(), encoding='unicode', method='xml')
        
        # Display result
        self.result_text.setText(result_str)
            
    def save_result(self):
        if not self.result_text.toPlainText():
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, '결과 파일 저장', '', 'XML Files (*.xml)')
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(self.result_text.toPlainText())

    def sync_scroll(self, value):
        # Get the sender of the signal
        sender = self.sender()
        
        # Prevent infinite loop
        if sender == self.original_text.verticalScrollBar():
            self.compare_text.verticalScrollBar().setValue(value)
        else:
            self.original_text.verticalScrollBar().setValue(value)

def main():
    app = QApplication(sys.argv)
    ex = FileCompareApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 