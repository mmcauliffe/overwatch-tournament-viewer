from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import cv2
import os
import json
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_dir)

from .settings import *

from .classes import *


class BrowserWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Overwatch annotator")

        # self.file_menu = QtGui.QMenu('&File', self)
        # self.file_menu.addAction('&Quit', self.fileQuit,
        #                         QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        # self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QHBoxLayout(self.main_widget)
        sc = AnnotationViewer()
        l.addWidget(sc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def center(self):
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


class AnnotationViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AnnotationViewer, self).__init__(parent)
        layout = QtWidgets.QHBoxLayout()
        self.frame = MainVideoFrameWidget()
        self.tree = TreeWidget()
        self.tree.select.currentItemChanged.connect(self.updateFile)
        self.annotator = AnnotationWidget()
        self.frame.timeChanged.connect(self.annotator.updateTimeLabel)
        self.annotator.updateTime.connect(self.frame.seekToTime)
        layout.addWidget(self.tree)
        layout.addWidget(self.frame)
        layout.addWidget(self.annotator)
        self.setLayout(layout)

    def updateFile(self):
        self.frame.setVideo(self.tree.value())
        self.annotator.updateMatch(self.tree.value())


class AnnotationWidget(QtWidgets.QWidget):
    updateTime = QtCore.pyqtSignal(int)
    players = QtCore.pyqtSignal(list, list)

    def __init__(self, parent=None):
        super(AnnotationWidget, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.timeLabel = QtWidgets.QLabel()
        self.annotationWidget = QtWidgets.QWidget()
        self.annotationWidget.setLayout(QtWidgets.QVBoxLayout())
        self.addGameButton = QtWidgets.QPushButton('Add game')
        self.addGameButton.clicked.connect(self.addGame)
        layout.addWidget(self.timeLabel)
        layout.addWidget(self.annotationWidget)
        layout.addWidget(self.addGameButton)
        self.setLayout(layout)
        self.annotations = None
        self.match = None

    def minimumSizeHint(self):
        return QtCore.QSize(400, 780)

    def addGame(self):
        game_count = self.annotationWidget.layout().count()
        if game_count == 0:
            lastID = 0
        else:
            w = self.annotationWidget.layout().itemAt(game_count - 1).widget()
            lastID = w.id
        w = GameWidget(lastID, Game(''))
        w.updateTime.connect(self.updateTime.emit)
        w.gameShown.connect(self.manageCollapsing)
        self.annotationWidget.layout().addWidget(w)

    def updateMatch(self, m):
        self.match = m
        self.games = parse_match(m)
        print('loaded match info')
        for i in reversed(range(self.annotationWidget.layout().count())):
            w = self.annotationWidget.layout().itemAt(i).widget()
            self.annotationWidget.layout().removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        for i, g in enumerate(self.games):
            w = GameWidget(i, g)
            w.updateTime.connect(self.updateTime.emit)
            w.gameShown.connect(self.manageCollapsing)
            self.annotationWidget.layout().addWidget(w)

    def manageCollapsing(self):
        for i in reversed(range(self.annotationWidget.layout().count())):
            w = self.annotationWidget.layout().itemAt(i).widget()
            if self.sender() == w:
                continue
            w.setChecked(False)
            w.showHide()

    def updateTimeLabel(self, time):
        self.timeLabel.setText(str(time))


class TeamWidget(QtWidgets.QWidget):
    switchHero = QtCore.pyqtSignal(int, str, str, int)
    killHero = QtCore.pyqtSignal(int, int, str, object, str)
    reviveHero = QtCore.pyqtSignal(int, int, str, int, str)
    heroDeath = QtCore.pyqtSignal(int, int, str)
    ultGain = QtCore.pyqtSignal(int, int, str)
    ultUse = QtCore.pyqtSignal(int, int, str)

    def __init__(self, side, parent=None):
        super(TeamWidget, self).__init__(parent)
        self.team = None
        self.side = side
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        layout = QtWidgets.QVBoxLayout()
        teamlayout = QtWidgets.QHBoxLayout()
        self.teamNameWidget = QtWidgets.QLineEdit()
        self.teamIDWidget = QtWidgets.QLineEdit()
        teamlayout.addWidget(QtWidgets.QLabel('Team name:'))
        teamlayout.addWidget(self.teamNameWidget)
        teamlayout.addWidget(QtWidgets.QLabel('Team ID:'))
        teamlayout.addWidget(self.teamIDWidget)
        self.teamColorWidget = QtWidgets.QLineEdit()
        teamlayout.addWidget(QtWidgets.QLabel('Team color:'))
        teamlayout.addWidget(self.teamColorWidget)
        self.playerlayout = QtWidgets.QHBoxLayout()
        for i in range(6):
            w = PlayerWidget(side, i)
            w.switchHero.connect(self.switchHero.emit)
            w.killHero.connect(self.killHero.emit)
            w.reviveHero.connect(self.reviveHero.emit)
            w.heroDeath.connect(self.heroDeath.emit)
            w.ultGain.connect(self.ultGain.emit)
            w.ultUse.connect(self.ultUse.emit)
            self.playerlayout.addWidget(w)
        layout.addLayout(teamlayout)
        layout.addLayout(self.playerlayout)
        self.setLayout(layout)

    def updateTime(self, time):
        for i in range(6):
            w = self.playerlayout.itemAt(i).widget()
            w.updateTime(time)

    def name(self):
        return self.teamNameWidget.text()

    def id(self):
        return self.teamIDWidget.text()

    def color(self):
        return self.teamColorWidget.text()

    def playerName(self, index):
        w = self.playerlayout.itemAt(index).widget()
        return w.nameEdit.text()

    def playerID(self, index):
        w = self.playerlayout.itemAt(index).widget()
        return w.idEdit.text()

    def updateRound(self, team):
        self.team = team
        self.teamNameWidget.setText(team.name)
        self.teamIDWidget.setText(team.id)
        self.teamColorWidget.setText(team.color)
        for i in range(6):
            w = self.playerlayout.itemAt(i).widget()
            w.setPlayer(self.team.players[i])


class AnnotationToolBar(QtWidgets.QWidget):
    switchHero = QtCore.pyqtSignal(int, str, str, int)
    killHero = QtCore.pyqtSignal(int, int, str, object, str)
    reviveHero = QtCore.pyqtSignal(int, int, str, int, str)
    heroDeath = QtCore.pyqtSignal(int, int, str)
    ultGain = QtCore.pyqtSignal(int, int, str)
    ultUse = QtCore.pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        super(AnnotationToolBar, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.leftTeamWidget = TeamWidget('left')
        self.leftTeamWidget.switchHero.connect(self.switchHero.emit)
        self.leftTeamWidget.killHero.connect(self.killHero.emit)
        self.leftTeamWidget.reviveHero.connect(self.reviveHero.emit)
        self.leftTeamWidget.heroDeath.connect(self.heroDeath.emit)
        self.leftTeamWidget.ultGain.connect(self.ultGain.emit)
        self.leftTeamWidget.ultUse.connect(self.ultUse.emit)
        layout.addWidget(self.leftTeamWidget)
        self.rightTeamWidget = TeamWidget('right')
        self.rightTeamWidget.switchHero.connect(self.switchHero.emit)
        self.rightTeamWidget.killHero.connect(self.killHero.emit)
        self.rightTeamWidget.reviveHero.connect(self.reviveHero.emit)
        self.rightTeamWidget.heroDeath.connect(self.heroDeath.emit)
        self.rightTeamWidget.ultGain.connect(self.ultGain.emit)
        self.rightTeamWidget.ultUse.connect(self.ultUse.emit)
        layout.addWidget(self.rightTeamWidget)
        self.setLayout(layout)

    def updateTeams(self, round_object):
        self.leftTeamWidget.updateTeam(round_object.left_performances)
        self.rightTeamWidget.updateTeam(round_object.right_performances)

    def updateTime(self, time):
        self.leftTeamWidget.updateTime(time)
        self.rightTeamWidget.updateTime(time)

    def getTime(self):
        time = self.parent().currentTime()
        return time

    def getPlayers(self, team):
        if team == 'right':
            return self.rightTeamWidget.team
        return self.leftTeamWidget.team


class AddKillDialog(QtWidgets.QDialog):
    def __init__(self, player, players, parent=None):
        super(AddKillDialog, self).__init__(parent)
        layout = QtWidgets.QFormLayout()
        self.playerKilled = QtWidgets.QComboBox()
        for p in players:
            self.playerKilled.addItem('{} ({})'.format(p.name, p.hero_at_time(self.parent().getTime())))
        self.playerKilled.addItem('mech')
        self.playerKilled.addItem('turret')
        self.playerKilled.addItem('supercharger')
        layout.addRow('Player killed', self.playerKilled)
        self.methodBox = QtWidgets.QComboBox()
        hero = player.hero_at_time(self.parent().getTime())
        if hero:
            for a in damaging_abilities[hero]:
                self.methodBox.addItem(a)
        layout.addRow('Method', self.methodBox)
        self.acceptButton = QtWidgets.QPushButton('Ok')
        self.acceptButton.clicked.connect(self.accept)
        layout.addWidget(self.acceptButton)
        self.setLayout(layout)

    def text(self):
        return self.playerKilled.currentText()

    def method(self):
        return self.methodBox.currentText()

    def value(self):
        return self.playerKilled.currentIndex()


class AddReviveDialog(QtWidgets.QDialog):
    def __init__(self, player, players, parent=None):
        super(AddReviveDialog, self).__init__(parent)
        layout = QtWidgets.QFormLayout()
        self.player = player
        self.playerKilled = QtWidgets.QComboBox()
        self.players = players
        for p in players:
            self.playerKilled.addItem('{} ({})'.format(p.name, p.hero_at_time(self.parent().getTime())))
        layout.addRow('Player killed', self.playerKilled)
        self.methodBox = QtWidgets.QComboBox()
        hero = player.hero_at_time(self.parent().getTime())
        if hero:
            for a in revive_abilities[hero]:
                self.methodBox.addItem(a)
        layout.addRow('Method', self.methodBox)
        self.acceptButton = QtWidgets.QPushButton('Ok')
        self.acceptButton.clicked.connect(self.accept)
        layout.addWidget(self.acceptButton)
        self.setLayout(layout)

    def accept(self):
        if not self.method():
            return
        if self.player.name == self.players[self.value()].name:
            return
        super(AddReviveDialog, self).accept()

    def method(self):
        return self.methodBox.currentText()

    def value(self):
        return self.playerKilled.currentIndex()


class PlayerWidget(QtWidgets.QGroupBox):
    switchHero = QtCore.pyqtSignal(int, str, str, int)
    killHero = QtCore.pyqtSignal(int, int, str, object, str)
    reviveHero = QtCore.pyqtSignal(int, int, str, int, str)
    heroDeath = QtCore.pyqtSignal(int, int, str)
    ultGain = QtCore.pyqtSignal(int, int, str)
    ultUse = QtCore.pyqtSignal(int, int, str)

    def __init__(self, team, id, parent=None):
        self.team = team
        self.id = id
        super(PlayerWidget, self).__init__('Player {}'.format(id + 1), parent)
        layout = QtWidgets.QVBoxLayout()
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        infolayout = QtWidgets.QHBoxLayout()
        self.heroBox = QtWidgets.QLabel()
        infolayout.addWidget(self.heroBox)
        switchlayout = QtWidgets.QFormLayout()
        self.player = None
        self.nameEdit = QtWidgets.QLineEdit()
        self.nameEdit.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.idEdit = QtWidgets.QLineEdit()
        self.idEdit.setFocusPolicy(QtCore.Qt.NoFocus)
        switchlayout.addRow('Name', self.nameEdit)
        switchlayout.addRow('ID', self.idEdit)
        self.switchBox = QtWidgets.QComboBox()
        self.switchBox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.switchBox.addItem('')
        for h in heroes:
            self.switchBox.addItem(h)
        buttonLayout = QtWidgets.QGridLayout()
        self.switchButton = QtWidgets.QPushButton('Add switch')
        self.switchButton.clicked.connect(self.addSwitch)
        self.switchButton.setFocusPolicy(QtCore.Qt.NoFocus)
        switchlayout.addRow('Switch', self.switchBox)
        infolayout.addLayout(switchlayout)
        layout.addLayout(infolayout)
        buttonLayout.addWidget(self.switchButton, 0, 0)
        self.setLayout(layout)
        self.addKillButton = QtWidgets.QPushButton('Add kill')
        self.addKillButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.addKillButton.clicked.connect(self.addKill)
        buttonLayout.addWidget(self.addKillButton, 0, 1)
        self.addDeathButton = QtWidgets.QPushButton('Add death')
        self.addDeathButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.addDeathButton.clicked.connect(self.addDeath)
        buttonLayout.addWidget(self.addDeathButton, 0, 2)
        self.addReviveButton = QtWidgets.QPushButton('Add revive')
        self.addReviveButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.addReviveButton.clicked.connect(self.addRevive)
        buttonLayout.addWidget(self.addReviveButton, 1, 0)
        self.addUltGainButton = QtWidgets.QPushButton('Add ult gain')
        self.addUltGainButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.addUltGainButton.clicked.connect(self.addUltGain)
        buttonLayout.addWidget(self.addUltGainButton, 1, 1)
        self.addUltUseButton = QtWidgets.QPushButton('Add ult use')
        self.addUltUseButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.addUltUseButton.clicked.connect(self.addUltUse)
        buttonLayout.addWidget(self.addUltUseButton, 1, 2)
        layout.addLayout(buttonLayout)

    def updateTime(self, time):
        if self.player is None:
            return

        frame = self.parent().parent().parent().frame.frame
        side = self.parent().side
        frame = box(frame, side, self.id)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.heroBox.setPixmap(pix)
        hero = self.player.hero_at_time(time)
        self.switchBox.setCurrentIndex(self.switchBox.findText(hero))
        self.addReviveButton.setEnabled(False)
        if hero == 'mercy':
            self.addReviveButton.setEnabled(True)
        has_ult = self.player.has_ult_at_time(time)
        if has_ult:
            self.addUltGainButton.setEnabled(False)
            self.addUltUseButton.setEnabled(True)
        else:
            self.addUltGainButton.setEnabled(True)
            self.addUltUseButton.setEnabled(False)

    def addKill(self):
        if self.team == 'left':
            opposing_players = self.parent().parent().getPlayers('right').players
        else:
            opposing_players = self.parent().parent().getPlayers('left').players
        dialog = AddKillDialog(self.player, opposing_players, self)
        if dialog.exec_():
            if dialog.text() in ['mech', 'turret', 'supercharger']:
                value = dialog.text()
            else:
                value = dialog.value()
            self.killHero.emit(self.getTime(), self.id, self.team, value, dialog.method())

    def addRevive(self):
        if self.team == 'left':
            team_players = self.parent().parent().getPlayers('left').players
        else:
            team_players = self.parent().parent().getPlayers('right').players
        dialog = AddReviveDialog(self.player, team_players, self)
        if dialog.exec_():
            self.reviveHero.emit(self.getTime(), self.id, self.team, dialog.value(), dialog.method())

    def addDeath(self):
        self.heroDeath.emit(self.getTime(), self.id, self.team)

    def addUltGain(self):
        self.ultGain.emit(self.getTime(), self.id, self.team)
        self.addUltGainButton.setEnabled(False)
        self.addUltUseButton.setEnabled(True)

    def addUltUse(self):
        self.ultUse.emit(self.getTime(), self.id, self.team)
        self.addUltGainButton.setEnabled(True)
        self.addUltUseButton.setEnabled(False)

    def getTime(self):
        time = self.parent().parent().getTime()
        return time

    def setPlayer(self, player):
        self.player = player
        if player.name is None:
            name = ''
        else:
            name = player.name
        self.nameEdit.setText(name)
        time = self.getTime()
        self.updateTime(time)

    def addSwitch(self):
        h = self.switchBox.currentText()
        self.addReviveButton.setEnabled(False)
        if h == 'mercy':
            self.addReviveButton.setEnabled(True)
        self.switchHero.emit(self.getTime(), h, self.team, self.id)


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int)

    def mousePressEvent(self, e):
        self.clicked.emit(int(self.text()))


class TimePointWidget(QtWidgets.QWidget):
    def __init__(self, initial_value, button_text, parent=None):
        super(TimePointWidget, self).__init__(parent)
        layout = QtWidgets.QHBoxLayout()
        self.label = ClickableLabel(initial_value)
        self.button = QtWidgets.QPushButton(button_text)
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)


class GameWidget(QtWidgets.QGroupBox):
    updateTime = QtCore.pyqtSignal(int)
    gameShown = QtCore.pyqtSignal()

    def __init__(self, id, game, parent=None):
        id += 1
        super(GameWidget, self).__init__('Game {}'.format(id), parent)
        self.id = id
        self.game = game
        self.setCheckable(True)
        self.setChecked(False)
        if id == 1:
            self.setChecked(True)
        self.toggled.connect(self.showHide)
        layout = QtWidgets.QVBoxLayout()
        self.mapWidget = QtWidgets.QComboBox()
        self.mapWidget.addItem('')
        for m in maps:
            self.mapWidget.addItem(m)
        self.mapWidget.setCurrentIndex(self.mapWidget.findText(game.map))
        self.mapWidget.currentIndexChanged.connect(self.setMap)
        layout.addWidget(self.mapWidget)
        self.gameBegin = TimePointWidget(str(game.begin), 'Update begin')
        self.gameEnd = TimePointWidget(str(game.end), 'Update end')

        self.gameBegin.label.clicked.connect(self.updateTime)
        self.gameEnd.label.clicked.connect(self.updateTime)
        self.gameBegin.button.clicked.connect(self.updateBegin)
        self.gameEnd.button.clicked.connect(self.updateEnd)

        layout.addWidget(self.gameBegin)
        layout.addWidget(self.gameEnd)
        for i, r in enumerate(game.rounds):
            w = RoundWidget(i, r)
            w.updateTime.connect(self.updateTime.emit)
            w.roundShown.connect(self.manageCollapsing)
            layout.addWidget(w)
        self.addRoundButton = QtWidgets.QPushButton('Add round')
        self.addRoundButton.clicked.connect(self.addRound)
        layout.addWidget(self.addRoundButton)
        self.setLayout(layout)
        self.showHide()

    def setMap(self):
        self.game.map = self.mapWidget.currentText()

    def updateBegin(self):
        new_begin = self.parent().parent().timeLabel.text()
        self.game.begin = int(new_begin)
        self.gameBegin.label.setText(new_begin)

    def updateEnd(self):
        new_end = self.parent().parent().timeLabel.text()
        self.game.end = int(new_end)
        self.gameEnd.label.setText(new_end)

    def addRound(self):
        round_count = self.layout().count() - 4
        if round_count <= 0:
            lastID = 0
        else:
            w = self.layout().itemAt(self.layout().count() - 2).widget()
            lastID = w.id
        team1 = Team('', '', '')
        team1.players = [Player('', '') for x in range(6)]
        team2 = Team('', '', '')
        team2.players = [Player('', '') for x in range(6)]
        w = RoundWidget(lastID, Round(team1, team2))
        w.updateTime.connect(self.updateTime.emit)
        w.roundShown.connect(self.manageCollapsing)
        self.layout().insertWidget(self.layout().count() - 1, w)

    def showHide(self):
        if self.isChecked():
            for i in reversed(range(self.layout().count())):
                self.layout().itemAt(i).widget().show()
            self.gameShown.emit()
        else:
            for i in reversed(range(self.layout().count())):
                w = self.layout().itemAt(i).widget()
                if isinstance(w, RoundWidget):
                    w.setChecked(False)
                    w.showHide()
                w.hide()

    def manageCollapsing(self):
        for i in reversed(range(self.layout().count())):
            w = self.layout().itemAt(i).widget()
            if self.sender() == w:
                continue
            if not isinstance(w, RoundWidget):
                continue
            w.setChecked(False)
            w.showHide()


class RoundDialog(QtWidgets.QDialog):
    def __init__(self, round_object, cap, parent=None):
        super(RoundDialog, self).__init__(parent)
        self.round_object = round_object

        layout = QtWidgets.QHBoxLayout()
        self.frame = RoundVideoFrameWidget(cap, round_object.begin, round_object.end)
        self.frame.toolBar.updateTeams(round_object)
        self.frame.toolBar.switchHero.connect(self.addSwitchEvent)
        self.frame.toolBar.killHero.connect(self.addKillEvent)
        self.frame.toolBar.reviveHero.connect(self.addReviveEvent)
        self.frame.toolBar.heroDeath.connect(self.addDeathEvent)
        self.frame.toolBar.ultGain.connect(self.addUltGainEvent)
        self.frame.toolBar.ultUse.connect(self.addUltUseEvent)
        layout.addWidget(self.frame)
        self.roundWidget = RoundWidget(None, round_object, dialog=True)
        self.roundWidget.updateTime.connect(self.frame.seekToTime)
        self.roundWidget.save.connect(self.accept)
        layout.addWidget(self.roundWidget)
        self.setLayout(layout)

    def accept(self):
        self.round_object.left_team.name = self.frame.toolBar.leftTeamWidget.name()
        self.round_object.left_team.id = self.frame.toolBar.leftTeamWidget.id()
        self.round_object.left_team.color = self.frame.toolBar.leftTeamWidget.color()
        for i in range(6):
            self.round_object.left_team.players[i].name = self.frame.toolBar.leftTeamWidget.playerName(i)
            self.round_object.left_team.players[i].id = self.frame.toolBar.leftTeamWidget.playerID(i)

        self.round_object.right_team.name = self.frame.toolBar.rightTeamWidget.name()
        self.round_object.right_team.id = self.frame.toolBar.rightTeamWidget.id()
        self.round_object.right_team.color = self.frame.toolBar.rightTeamWidget.color()
        for i in range(6):
            self.round_object.right_team.players[i].name = self.frame.toolBar.rightTeamWidget.playerName(i)
            self.round_object.right_team.players[i].id = self.frame.toolBar.rightTeamWidget.playerID(i)
        super(RoundDialog, self).accept()

    def currentTime(self):
        return self.frame.currentTime()

    def addSwitchEvent(self, time_point, new_hero, team, i):
        if team == 'left':
            self.round_object.left_team.players[i].add_switch(time_point, new_hero)
        else:
            self.round_object.right_team.players[i].add_switch(time_point, new_hero)
        self.roundWidget.refreshList()

    def addKillEvent(self, time_point, i, team, killed_i, method):
        if team == 'left':
            self.round_object.left_team.players[i].kills.append((time_point, killed_i, method))
            if isinstance(killed_i, int):
                self.round_object.right_team.players[killed_i].add_death(time_point)
        else:
            self.round_object.right_team.players[i].kills.append((time_point, killed_i, method))
            if isinstance(killed_i, int):
                self.round_object.left_team.players[killed_i].add_death(time_point)
        self.roundWidget.refreshList()

    def addReviveEvent(self, time_point, i, team, revived_i, method):
        if team == 'left':
            self.round_object.left_team.players[i].revives.append((time_point, revived_i, method))
        else:
            self.round_object.right_team.players[i].revives.append((time_point, revived_i, method))
        self.roundWidget.refreshList()

    def addDeathEvent(self, time_point, i, team):
        if team == 'left':
            self.round_object.left_team.players[i].add_death(time_point)
        else:
            self.round_object.right_team.players[i].add_death(time_point)
        self.roundWidget.refreshList()

    def addUltGainEvent(self, time_point, i, team):
        if team == 'left':
            self.round_object.left_team.players[i].add_ult_gain(time_point)
        else:
            self.round_object.right_team.players[i].add_ult_gain(time_point)
        self.roundWidget.refreshList()

    def addUltUseEvent(self, time_point, i, team):
        if team == 'left':
            self.round_object.left_team.players[i].add_ult_use(time_point)
        else:
            self.round_object.right_team.players[i].add_ult_use(time_point)
        self.roundWidget.refreshList()


class RoundWidget(QtWidgets.QGroupBox):
    updateTime = QtCore.pyqtSignal(int)
    roundShown = QtCore.pyqtSignal()
    save = QtCore.pyqtSignal()

    def __init__(self, id, round_object, dialog=False, parent=None):
        if dialog:
            id = 'current'
        else:
            id += 1
        super(RoundWidget, self).__init__('Round {}'.format(id), parent)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        layout = QtWidgets.QVBoxLayout()

        self.setLayout(layout)
        self.round_object = round_object

        self.id = id
        self.list = QtWidgets.QListWidget()
        if not dialog:
            self.setCheckable(True)
            self.setChecked(False)
            if id == 1:
                self.setChecked(True)
            self.toggled.connect(self.showHide)
            self.roundBegin = TimePointWidget(str(round_object.begin), 'Update begin')
            self.roundEnd = TimePointWidget(str(round_object.end), 'Update end')

            self.roundBegin.label.clicked.connect(self.updateTime)
            self.roundEnd.label.clicked.connect(self.updateTime)
            self.roundBegin.button.clicked.connect(self.updateBegin)
            self.roundEnd.button.clicked.connect(self.updateEnd)

            self.attackingWidget = QtWidgets.QComboBox()
            self.attackingWidget.addItems(['', 'left', 'right'])
            self.attackingWidget.setCurrentIndex(self.attackingWidget.findText(getattr(self.round_object, 'attacking_side', '')))

            layout.addWidget(self.roundBegin)
            layout.addWidget(self.roundEnd)
            layout.addWidget(self.attackingWidget)
            self.editButton = QtWidgets.QPushButton('Edit round annotations')
            self.editButton.clicked.connect(self.editRound)
            layout.addWidget(self.editButton)
        else:
            self.getFromPreviousButton = QtWidgets.QPushButton('Get metadata from previous round')
            self.getFromPreviousButton.clicked.connect(self.getFromPrevious)
            layout.addWidget(self.getFromPreviousButton)
            self.deleteButton = QtWidgets.QPushButton('Delete all at selected timepoint')
            self.deleteButton.clicked.connect(self.deleteAnnotations)
            layout.addWidget(self.deleteButton)
            self.addAttackButton = QtWidgets.QPushButton('Add attack')
            if self.round_object.attacking_team is None:
                layout.addWidget(self.addAttackButton)
                self.addAttackButton.clicked.connect(self.addAttack)
            self.addPauseButton = QtWidgets.QPushButton('Add pause')
            self.addPauseButton.clicked.connect(self.addPause)
            layout.addWidget(self.addPauseButton)

            self.addUnpauseButton = QtWidgets.QPushButton('Add unpause')
            self.addUnpauseButton.clicked.connect(self.addUnpause)
            layout.addWidget(self.addUnpauseButton)
            self.addPointsButton = QtWidgets.QPushButton('Add points')
            self.addPointsButton.clicked.connect(self.addPoints)
            layout.addWidget(self.addPointsButton)
        layout.addWidget(self.list)
        if round_object is not None:
            for d in self.round_object.construct_timeline(absolute_time=not dialog):
                self.list.addItem(' '.join(map(str, d)))
        self.list.currentItemChanged.connect(self.newTime)
        if not dialog:
            self.showHide()
        else:
            self.saveButton = QtWidgets.QPushButton('Save round')
            self.saveButton.clicked.connect(self.save)
            layout.addWidget(self.saveButton)

    def getFromPrevious(self):
        currentRound_id = self.parent().parent().id
        if currentRound_id == 1:
            return
        index = currentRound_id + 1
        w = self.parent().parent().parent().layout().itemAt(index).widget()
        left_team = Team(w.round_object.left_team.name, w.round_object.left_team.id, w.round_object.left_team.color)
        right_team = Team(w.round_object.right_team.name, w.round_object.right_team.id, w.round_object.right_team.color)
        left_team.players = [Player(x.name, x.id) for x in w.round_object.left_team.players]
        right_team.players = [Player(x.name, x.id) for x in w.round_object.right_team.players]
        self.parent().frame.toolBar.updateTeams(left_team, right_team)

    def addAttack(self):
        dialog = QtWidgets.QDialog()
        dialog.teamWidget = QtWidgets.QComboBox()
        dialog.teamWidget.addItem('left')
        dialog.teamWidget.addItem('right')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(dialog.teamWidget)
        b = QtWidgets.QPushButton('Save')
        b.clicked.connect(dialog.accept)
        layout.addWidget(b)
        dialog.setLayout(layout)
        if dialog.exec_():
            self.round_object.round_events.append(
                (self.parent().currentTime(), 'ATTACK', dialog.teamWidget.currentText()))
            self.refreshList()

    def addPause(self):
        self.round_object.round_events.append((self.parent().currentTime(), 'PAUSE'))
        self.refreshList()

    def addUnpause(self):
        self.round_object.round_events.append((self.parent().currentTime(), 'UNPAUSE'))
        self.refreshList()

    def addPoints(self):
        dialog = QtWidgets.QDialog()
        dialog.teamWidget = QtWidgets.QComboBox()
        dialog.teamWidget.addItem('left')
        dialog.teamWidget.addItem('right')
        dialog.pointsWidget = QtWidgets.QLineEdit()
        layout = QtWidgets.QFormLayout()
        layout.addRow('Team', dialog.teamWidget)
        layout.addRow('Points', dialog.pointsWidget)
        b = QtWidgets.QPushButton('Save')
        b.clicked.connect(dialog.accept)
        layout.addWidget(b)
        dialog.setLayout(layout)
        if dialog.exec_():
            self.round_object.round_events.append((self.parent().currentTime(), 'POINTS',
                                                   dialog.teamWidget.currentText(), int(dialog.pointsWidget.text())))
            self.refreshList()

    def deleteAnnotations(self):
        t = self.list.currentItem().text().split()
        timepoint = int(t[0])
        self.round_object.remove_annotations(timepoint)
        self.refreshList()

    def refreshList(self):
        self.list.currentItemChanged.disconnect(self.newTime)
        self.list.clear()
        for d in self.round_object.construct_timeline(absolute_time=False):
            self.list.addItem(' '.join(map(str, d)))
        self.list.currentItemChanged.connect(self.newTime)

    def editRound(self):
        cap = self.parent().parent().parent().parent().frame.cap
        dialog = RoundDialog(self.round_object, cap, parent=self)
        if dialog.exec_():
            self.round_object = dialog.round_object
            match = self.getMatch()
            game_num = self.getGameNum()
            round_num = self.id
            meta_path = os.path.join(annotations_dir, match, str(game_num), 'meta.json')
            game = self.getGame()
            game.export(meta_path)
            path = os.path.join(annotations_dir, match, str(game_num), '{}_{}_data.txt'.format(game_num, round_num))
            self.round_object.export(path)

    def getGameNum(self):
        return self.parent().id

    def getGame(self):
        return self.parent().game

    def getMatch(self):
        return self.parent().parent().parent().match

    def updateBegin(self):
        new_begin = self.parent().parent().parent().timeLabel.text()
        self.round_object.begin = int(new_begin)
        self.roundBegin.label.setText(new_begin)

    def updateEnd(self):
        new_end = self.parent().parent().parent().timeLabel.text()
        self.round_object.end = int(new_end)
        self.roundEnd.label.setText(new_end)

    def showHide(self):
        if self.isChecked():
            for i in reversed(range(self.layout().count())):
                self.layout().itemAt(i).widget().show()
            self.roundShown.emit()
            if self.round_object.begin != 0:
                self.updateTime.emit(self.round_object.begin)
        else:
            for i in reversed(range(self.layout().count())):
                w = self.layout().itemAt(i).widget()
                if isinstance(w, RoundWidget):
                    w.setChecked(False)
                    w.showHide()
                w.hide()

    def newTime(self):
        time = self.list.currentItem().text().split()[0]
        self.updateTime.emit(int(time))


class TreeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TreeWidget, self).__init__(parent)
        self.select = QtWidgets.QListWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.select)
        for f in test_files:
            self.select.addItem(str(f))
        self.setLayout(layout)

    def value(self):
        return self.select.currentItem().text()


class ImageWidget(QtWidgets.QLabel):
    def sizeHint(self):
        return QtCore.QSize(1280, 720)

    def updateImage(self, frame):
        self.frame = frame
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.setPixmap(pix)


class MainVideoFrameWidget(QtWidgets.QWidget):
    timeChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(MainVideoFrameWidget, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.cap = None
        self.fps = 0
        layout = QtWidgets.QVBoxLayout()
        self.frame = ImageWidget()

        self.timeline = TimeLineWidget()
        layout.addWidget(self.frame)
        layout.addWidget(self.timeline)
        self.timeline.timeChanged.connect(self.seek)

        self.setLayout(layout)
        self.vid_pos = 0

    def setVideo(self, filename):
        self.vid_pos = 0
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        directory = os.path.join(annotations_dir, filename)
        vod_path = os.path.join(directory, '{}.mp4'.format(filename))
        events_path = os.path.join(directory, 'events.txt')
        self.cap = cv2.VideoCapture(vod_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.nextFrameSlot()
        self.duration = self.num_frames / self.fps

    def seekToTime(self, time):
        if self.cap is not None:
            self.vid_pos = int(time * self.fps)
            self.cap.set(1, self.vid_pos)
            self.nextFrameSlot()

    def currentTime(self):
        return int(self.vid_pos / self.fps)

    def seek(self, time_percent):
        if self.cap is not None:
            self.vid_pos = int(time_percent * self.num_frames)
            self.cap.set(1, self.vid_pos)
            self.nextFrameSlot()

    def nextFrameSlot(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            self.frame.updateImage(frame)
            if self.timeline is not None:
                self.timeline.updateTime(self.vid_pos / self.num_frames)
            self.vid_pos += 1
            self.timeChanged.emit(self.currentTime())

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000.0 / self.fps))

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        if self.cap is not None:
            self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_D, QtCore.Qt.Key_Right]:
            change = 2
        elif e.key() in [QtCore.Qt.Key_A, QtCore.Qt.Key_Left]:
            change = -2
        elif e.key() in [QtCore.Qt.Key_W, QtCore.Qt.Key_Up]:
            change = int(self.fps)
        elif e.key() in [QtCore.Qt.Key_S, QtCore.Qt.Key_Down]:
            change = -1 * int(self.fps)
        else:
            print(e.key())
            return
        self.vid_pos += change
        if self.vid_pos < 0:
            self.vid_pos = 0
        if self.vid_pos > self.num_frames - 1:
            self.vid_pos = self.num_frames - 1
        self.cap.set(1, self.vid_pos)
        self.nextFrameSlot()


class RoundVideoFrameWidget(MainVideoFrameWidget):
    timeChanged = QtCore.pyqtSignal(int)

    def __init__(self, cap, min_time, max_time, parent=None):
        super(MainVideoFrameWidget, self).__init__(parent)
        self.timeline = None
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.cap = cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.min_time = min_time
        self.min_frame = (min_time * self.fps)
        self.max_time = max_time
        self.max_frame = (max_time * self.fps)
        layout = QtWidgets.QVBoxLayout()
        self.frame = ImageWidget()

        layout.addWidget(self.frame)
        self.toolBar = AnnotationToolBar()
        self.timeChanged.connect(self.toolBar.updateTime)
        layout.addWidget(self.toolBar)
        self.setLayout(layout)
        self.vid_pos = int(self.min_time * self.fps)
        self.cap.set(1, self.vid_pos)
        self.nextFrameSlot()

    def seekToTime(self, time):
        if self.cap is not None:
            time += self.min_time
            self.vid_pos = int(time * self.fps)
            self.cap.set(1, self.vid_pos)
            self.nextFrameSlot()

    def currentTime(self):
        return int(self.vid_pos / self.fps) - self.min_time

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_D, QtCore.Qt.Key_Right]:
            change = 2
        elif e.key() in [QtCore.Qt.Key_A, QtCore.Qt.Key_Left]:
            change = -2
        elif e.key() in [QtCore.Qt.Key_W, QtCore.Qt.Key_Up]:
            change = int(self.fps)
        elif e.key() in [QtCore.Qt.Key_S, QtCore.Qt.Key_Down]:
            change = -1 * int(self.fps)
        else:
            print(e.key())
            return
        self.vid_pos += change
        if self.vid_pos < self.min_frame:
            self.vid_pos = self.min_frame
        if self.vid_pos > self.max_frame - 1:
            self.vid_pos = self.max_frame - 1
        self.cap.set(1, self.vid_pos)
        self.nextFrameSlot()

    def deleteLater(self):
        QtWidgets.QWidget.deleteLater(self)


class TimeLineWidget(QtWidgets.QGraphicsView):
    timeChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super(TimeLineWidget, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QGraphicsView.NoFrame)
        self.graphics_scene = QtWidgets.QGraphicsScene()
        self.graphics_scene.setSceneRect(0, 0, 1280, 50)
        self.setScene(self.graphics_scene)

    def updateTime(self, time_percent):
        self.graphics_scene.clear()
        pen = QtGui.QPen(QtGui.QColor('red'))
        x = int(1280 * time_percent)

        self.graphics_scene.addLine(x, 0, x, 100, pen)

    def sizeHint(self):
        return QtCore.QSize(1280, 50)

    def mousePressEvent(self, e):
        self.timeChanged.emit(e.x() / 1280)


if __name__ == '__main__':
    import sys

    sys._excepthook = sys.excepthook


    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook
    app = QtWidgets.QApplication(sys.argv)
    window = BrowserWindow()
    window.show()
    window.center()
    sys.exit(app.exec_())
