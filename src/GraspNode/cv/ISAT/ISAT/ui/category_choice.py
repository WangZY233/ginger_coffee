# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'category_choice.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(350, 399)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        Dialog.setFont(font)
        Dialog.setWindowTitle("")
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.widget_3 = QtWidgets.QWidget(Dialog)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.widget_3)
        self.label_2.setMinimumSize(QtCore.QSize(60, 0))
        self.label_2.setMaximumSize(QtCore.QSize(60, 16777215))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.lineEdit_category = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_category.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_category.setReadOnly(True)
        self.lineEdit_category.setObjectName("lineEdit_category")
        self.horizontalLayout_3.addWidget(self.lineEdit_category)
        self.label = QtWidgets.QLabel(self.widget_3)
        self.label.setMinimumSize(QtCore.QSize(50, 0))
        self.label.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.lineEdit_group = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_group.setMinimumSize(QtCore.QSize(60, 0))
        self.lineEdit_group.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_group.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_group.setText("")
        self.lineEdit_group.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_group.setObjectName("lineEdit_group")
        self.horizontalLayout_3.addWidget(self.lineEdit_group)
        self.verticalLayout.addWidget(self.widget_3)
        self.widget_5 = QtWidgets.QWidget(Dialog)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.widget_5)
        self.label_3.setMinimumSize(QtCore.QSize(60, 0))
        self.label_3.setMaximumSize(QtCore.QSize(60, 16777215))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.lineEdit_note = QtWidgets.QLineEdit(self.widget_5)
        self.lineEdit_note.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_note.setObjectName("lineEdit_note")
        self.horizontalLayout_5.addWidget(self.lineEdit_note)
        self.label_4 = QtWidgets.QLabel(self.widget_5)
        self.label_4.setMinimumSize(QtCore.QSize(50, 0))
        self.label_4.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.label_layer = QtWidgets.QLabel(self.widget_5)
        self.label_layer.setMinimumSize(QtCore.QSize(60, 0))
        self.label_layer.setMaximumSize(QtCore.QSize(60, 16777215))
        self.label_layer.setText("")
        self.label_layer.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_layer.setObjectName("label_layer")
        self.horizontalLayout_5.addWidget(self.label_layer)
        self.verticalLayout.addWidget(self.widget_5)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(Dialog)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_iscrowded = QtWidgets.QCheckBox(self.widget_2)
        self.checkBox_iscrowded.setObjectName("checkBox_iscrowded")
        self.horizontalLayout_2.addWidget(self.checkBox_iscrowded)
        spacerItem = QtWidgets.QSpacerItem(97, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.pushButton_cancel = QtWidgets.QPushButton(self.widget_2)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/关闭_close-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_cancel.setIcon(icon)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.horizontalLayout_2.addWidget(self.pushButton_cancel)
        self.pushButton_apply = QtWidgets.QPushButton(self.widget_2)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/icons/校验_check-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_apply.setIcon(icon1)
        self.pushButton_apply.setObjectName("pushButton_apply")
        self.horizontalLayout_2.addWidget(self.pushButton_apply)
        self.verticalLayout.addWidget(self.widget_2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        self.label_2.setText(_translate("Dialog", "category:"))
        self.label.setText(_translate("Dialog", "group:"))
        self.lineEdit_group.setPlaceholderText(_translate("Dialog", "group id"))
        self.label_3.setText(_translate("Dialog", "note:"))
        self.lineEdit_note.setPlaceholderText(_translate("Dialog", "add extra note here"))
        self.label_4.setText(_translate("Dialog", "layer:"))
        self.checkBox_iscrowded.setText(_translate("Dialog", "is crowded"))
        self.pushButton_cancel.setText(_translate("Dialog", "cancel"))
        self.pushButton_apply.setText(_translate("Dialog", "apply"))
# import icons_rc
