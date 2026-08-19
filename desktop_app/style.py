"""AITIC Desktop 视觉规范。"""

APP_STYLE = r"""
QWidget {
    color: #172b46;
    font-size: 12px;
}
QMainWindow, QDialog, QWidget#ContentRoot { background: #ffffff; }
QFrame#Sidebar { background: #f7f8fa; border: 0; border-right: 1px solid #e2e6eb; }
QLabel#BrandMark {
    background: #ffffff; color: #101922; border: 2px solid #cbd2da;
    border-radius: 10px; font-size: 19px; font-weight: 800;
}
QLabel#Brand { color: #102b4d; font-size: 18px; font-weight: 650; }
QLabel#BrandSub { color: #768397; font-size: 11px; }
QLabel#PrivacyPill {
    background: #edf9f1; color: #137542; border: 1px solid #bfe5cb;
    border-radius: 12px; padding: 4px 10px; font-size: 11px; font-weight: 600;
}
QLabel#SidebarSection { color: #8290a3; font-size: 11px; padding: 8px 4px 0 4px; }
QLabel#SelectionSummary { color: #53647a; font-size: 11px; padding: 3px; }
QLabel#RuntimeState { color: #9a513b; font-size: 11px; padding: 3px; }
QLabel#RuntimeState[ready="true"] { color: #138154; }
QLabel#PageTitle { color: #102a4b; font-size: 23px; font-weight: 650; }
QLabel#PageDescription { color: #64758c; font-size: 12px; }
QLabel#SectionTitle { color: #182f4b; font-size: 14px; font-weight: 650; }
QLabel#Muted { color: #748399; }
QLabel#TopBadge {
    background: #ffffff; color: #1a416b; border: 1px solid #d2deeb;
    border-radius: 7px; padding: 7px 10px; font-weight: 550;
}
QPushButton#TopBadge {
    background: #ffffff; color: #1a416b; border: 1px solid #d2deeb;
    border-radius: 7px; padding: 7px 10px; font-weight: 550;
}
QPushButton#TopBadge:hover { background: #edf3f8; border-color: #b8cbe0; }
QPushButton#TopBadge:checked { background: #dfeaf5; border-color: #8eafd0; }
QLabel#PillOk { background: #dff7eb; color: #16734a; border-radius: 11px; padding: 3px 9px; }
QLabel#PillBad { background: #ffebe9; color: #a33a30; border-radius: 11px; padding: 3px 9px; }
QFrame#Card, QFrame#EvidencePanel {
    background: #ffffff; border: 1px solid #dce4ed; border-radius: 12px;
}
QFrame#MaterialBar {
    background: #f4f8fc; border: 1px solid #cfe0f0; border-radius: 11px;
}
QPushButton {
    background: #245f96; color: white; border: 1px solid #245f96;
    border-radius: 8px; padding: 8px 15px; font-weight: 600;
}
QPushButton:hover { background: #1b507f; border-color: #1b507f; }
QPushButton:pressed { background: #143f66; }
QPushButton:disabled { background: #7f9bb6; border-color: #7f9bb6; color: #e9f0f6; }
QPushButton[secondary="true"], QPushButton[chip="true"] {
    color: #294765; background: #ffffff; border: 1px solid #cfdae7;
}
QPushButton[secondary="true"]:hover, QPushButton[chip="true"]:hover { background: #edf3f8; }
QPushButton[chip="true"] { padding: 6px 11px; border-radius: 8px; }
QPushButton[danger="true"] { background: #c84639; border-color: #c84639; }
QPushButton#NewChatButton {
    color: #132c48; background: #eef0f3; border: 0; border-radius: 12px;
    padding: 12px 15px; text-align: left; font-size: 13px; font-weight: 600;
}
QPushButton#NewChatButton:hover { background: #e2e6eb; }
QPushButton#AskButton { min-width: 82px; padding: 10px 20px; border-radius: 10px; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {
    background: #ffffff; border: 1px solid #cfd9e5; border-radius: 8px;
    padding: 7px 9px; selection-background-color: #5b92e5;
}
QComboBox#CrispComboBox { padding-right: 30px; min-height: 22px; }
QComboBox#CrispComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right;
    width: 28px; border: 0; background: transparent;
}
QComboBox#CrispComboBox::down-arrow { image: none; width: 0; height: 0; }
QComboBox#CrispComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #cfd9e5; selection-background-color: #e5edf5;
    selection-color: #173a5e; padding: 4px; outline: 0;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #477eae;
}
QLineEdit#SessionSearch { background: #ffffff; border-radius: 9px; padding: 7px 9px; }
QTextEdit#Composer { border-radius: 16px; padding: 13px; font-size: 13px; }
QTextBrowser {
    background: #ffffff; border: 1px solid #dce4ed; border-radius: 11px; padding: 11px;
}
QTextBrowser#ChatView { border: 0; background: #ffffff; padding: 8px; }
QListWidget#Navigation, QListWidget#SessionList {
    background: transparent; color: #263f5d; border: 0; outline: 0;
}
QListWidget#Navigation::item { border-radius: 10px; padding: 11px 13px; margin: 2px 0; }
QListWidget#Navigation::item:hover { background: #edf0f4; color: #102a4b; }
QListWidget#Navigation::item:selected { background: #e1e6ec; color: #153d66; font-weight: 600; }
QListWidget#SessionList::item { padding: 7px 8px; border-radius: 7px; color: #354b64; }
QListWidget#SessionList::item:hover { background: #edf0f4; }
QListWidget#SessionList::item:selected { background: #dfe7ef; color: #173a5e; }
QToolButton#MoreButton {
    background: transparent; color: #263f5d; border: 0; border-radius: 10px;
    padding: 10px 13px; text-align: left; font-size: 12px; font-weight: 400;
}
QToolButton#MoreButton:hover, QToolButton#MoreButton:pressed { background: #e7ebef; }
QMenu { background: #ffffff; border: 1px solid #d9e1ea; padding: 5px; font-size: 12px; }
QMenu::item { padding: 7px 26px 7px 11px; border-radius: 5px; font-weight: 400; }
QMenu::item:selected { background: #e8eef4; color: #173d64; }
QTableWidget, QTreeWidget {
    background: #ffffff; alternate-background-color: #f8fafc;
    border: 1px solid #dce4ed; border-radius: 9px;
    gridline-color: #e7edf4; outline: 0;
}
QHeaderView::section {
    background: #f0f4f8; color: #405873; border: 0;
    border-right: 1px solid #dce4ed; border-bottom: 1px solid #dce4ed;
    padding: 8px; font-weight: 600;
}
QTabWidget::pane { border: 0; }
QTabBar::tab {
    background: transparent; color: #66778d; padding: 10px 16px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #245f96; border-bottom: 2px solid #245f96; font-weight: 600; }
QProgressBar { border: 0; background: #dce5ef; border-radius: 3px; height: 7px; }
QProgressBar::chunk { background: #397db7; border-radius: 3px; }
QStatusBar { background: #ffffff; border-top: 1px solid #e1e6ec; color: #62748b; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c3ccd7; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: #172b43; color: white; border: 0; padding: 5px; }
"""
