[app]
title = Gomoku
package.name = gomoku
package.domain = org.azat
source.dir = .
source.include_exts = py,png,jpg,kv,ttf,wav,mp3
version = 1.0
requirements = python3,kivy

# İkon koyacaksan:
# icon.filename = icon.png

[buildozer]
log_level = 2

[android]
android.archs = arm64-v8a

android.api = 33
android.minapi = 21

# KRİTİK: aidl build-tools içinden gelir
android.build_tools_version = 34.0.0

# SDK lisanslarını otomatik kabul et
android.accept_sdk_license = True


