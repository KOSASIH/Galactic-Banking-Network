// src/services/VRService.js

class VRService {
    static enterVR() {
        return new Promise((resolve, reject) => {
            if (navigator.getVRDisplays) {
                navigator.getVRDisplays().then(displays => {
                    if (displays.length > 0) {
                        const vrDisplay = displays[0];
                        vrDisplay.requestPresent([{ source: document.body }])
                            .then(() => resolve('Entered VR mode'))
                            .catch(err => reject('Failed to enter VR mode: ' + err));
                    } else {
                        reject('No VR displays found');
                    }
                });
            } else {
                reject('WebVR not supported');
            }
        });
    }

    static exitVR() {
        return new Promise((resolve, reject) => {
            if (navigator.getVRDisplays) {
                navigator.getVRDisplays().then(displays => {
                    if (displays.length > 0) {
                        const vrDisplay = displays[0];
                        vrDisplay.exitPresent()
                            .then(() => resolve('Exited VR mode'))
                            .catch(err => reject('Failed to exit VR mode: ' + err));
                    } else {
                        reject('No VR displays found');
                    }
                });
            } else {
                reject('WebVR not supported');
            }
        });
    }
}

export default VRService;
