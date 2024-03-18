# AppDaemon and Predbat

* AppDaemon is a loosely coupled, multithreaded, sandboxed Python
execution environment for writing automation apps for Home Assistant home
automation software. It also provides a configurable dashboard (HADashboard)
suitable for wall mounted tablets.

* Predbat is a Home battery prediction and automatic charging for Home Assistant

## Installation

* Go to settings, Add-ons, Add on store (bottom right), Top right three dots, Repositories and add 'https://github.com/springfall2008/appdaemon-predbat' to the list and close.
* Next select AppDaemon with Predbat and Install and then hit 'start'
* If you haven't already change your Editor settings to ensure 'enforce basepath' is disabled (settings, Add-ons, File Editor, Configuration)
* Now use your Editor to find '/addon_configs/46f69597_appdaemon-predbat', here there is
  * predbat.log - contains the active logfile with any errors
  * apps/apps.yaml - you need to edit apps.yaml to remove the template settings and customise
* Once you have edited apps.yaml click 'restart' on the appdaemon-predbat add-on
* Continue to set up Predbat as per the documentation

## Migration from Predbat installed with AppDaemon

* If you have a previous install of Predbat with AppDaemon then first
  * Take a copy of your existing apps.yaml
  * Either uninstall AppDaemon or delete the 'homeassistant/appdeamon/apps/batpred' directory from your system
* After installing appdaemon-predbat then copy your saved apps.yaml over the template apps.yaml in '/addon_configs/46f69597_appdaemon-predbat'
