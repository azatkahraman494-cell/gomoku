[app]
title = Gomoku
package.name = gomoku
package.domain = org.seninad
source.dir = .
source.include_exts = py,png,jpg,kv,ttf,wav,mp3
version = 1.0
requirements = python3,kivy

# İkon koyacaksan:
# icon.filename = icon.png

[buildozer]
log_level = 2

[android]
# Sadece modern Android cihazlar için (önerilen):
android.archs = arm64-v8a

# Eğer eski cihaz da istiyorsan:
# android.archs = arm64-v8a,armeabi-v7a

# Android API ayarı (genelde böyle bırakmak yeter):
android.api = 33
android.minapi = 21

# Eğer ileride izin gerekirse buraya:
# android.permissions = INTERNET
