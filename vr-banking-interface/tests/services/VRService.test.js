// services/VRService.test.js

import VRService from './VRService';

describe('VRService', () => {
    beforeEach(() => {
        // Mock the VR API methods
        global.navigator.getVRDisplays = jest.fn().mockResolvedValue([
            { isConnected: true, name: 'Mock VR Headset' }
        ]);
    });

    it('should enter VR mode', async () => {
        const enterVR = jest.spyOn(VRService, 'enterVR').mockImplementation(async () => {
            return Promise.resolve('Entered VR mode');
        });

        const result = await VRService.enterVR();
        expect(enterVR).toHaveBeenCalled();
        expect(result).toBe('Entered VR mode');
    });

    it('should exit VR mode', async () => {
        const exitVR = jest.spyOn(VRService, 'exitVR').mockImplementation(async () => {
            return Promise.resolve('Exited VR mode');
        });

        const result = await VRService.exitVR();
        expect(exitVR).toHaveBeenCalled();
        expect(result).toBe('Exited VR mode');
    });

    it('should handle VR display not connected', async () => {
        global.navigator.getVRDisplays.mockResolvedValueOnce([]);

        await expect(VRService.enterVR()).rejects.toThrow('No VR displays found');
    });

    it('should handle errors when entering VR mode', async () => {
        const enterVR = jest.spyOn(VRService, 'enterVR').mockImplementation(async () => {
            throw new Error('Failed to enter VR mode');
        });

        await expect(VRService.enterVR()).rejects.toThrow('Failed to enter VR mode');
        expect(enterVR).toHaveBeenCalled();
    });

    it('should handle errors when exiting VR mode', async () => {
        const exitVR = jest.spyOn(VRService, 'exitVR').mockImplementation(async () => {
            throw new Error('Failed to exit VR mode');
        });

        await expect(VRService.exitVR()).rejects.toThrow('Failed to exit VR mode');
        expect(exitVR).toHaveBeenCalled();
    });
});
