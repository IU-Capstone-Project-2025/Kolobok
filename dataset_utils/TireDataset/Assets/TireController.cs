using UnityEngine;
using System.Collections.Generic;
using System.Collections;
public class TireController : MonoBehaviour
{
    public List<CruchunVerchun> tires;

    public int currentTireIndex = 0;
    public Camera mainCamera;
    public float framesRate = 10;
    public int count = 10;
    public int lighting = 0;
    public int environment = 0;
    public string datasetPath = "Users/dmitry057/Projects/Kolobok/TireDataset/Dataset";
    private WaitForSeconds waitTime;
    private WaitForSeconds waitTimeBetweenTires = new WaitForSeconds(1f);
    private int currentCount = 0;
    private void Start(){
        waitTime = new WaitForSeconds(1f / framesRate); 
        StartCoroutine(CaptureTireImages());
        mainCamera.GetComponent<CameraOrbitalController>().center = tires[currentTireIndex].center;
        // Turn off all tires except the current one
        for (int i = 0; i < tires.Count; i++)
        {
            if (i != currentTireIndex)
            {
                tires[i].gameObject.SetActive(false);
            }
            else
            {
                tires[i].gameObject.SetActive(true);
            }
        }
    }
    public void CaptureTireImage(int count)
    {
        // Set camera resolution to 1920x1080
        RenderTexture renderTexture = new RenderTexture(1920, 1080, 24);
        mainCamera.targetTexture = renderTexture;
        
        // Render the camera view
        mainCamera.Render();
        
        // Create a texture to read the camera render
        RenderTexture.active = mainCamera.targetTexture;
        Texture2D screenshot = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0);
        screenshot.Apply();
        
        // Convert texture to bytes
        byte[] bytes = screenshot.EncodeToJPG();
        
        // Generate filename using tire name and current tire thin
        string filename = tires[currentTireIndex].name +"_"+tires[currentTireIndex].currentTireThin + "_" + currentCount + "_" + lighting + "_" + environment + ".jpg";
        currentCount++;
        // Save the image
        string filepath = datasetPath + "/" + filename;
        System.IO.File.WriteAllBytes(filepath, bytes);
        
        // Clean up
        RenderTexture.active = null;
        mainCamera.targetTexture = null;
        DestroyImmediate(screenshot);
        DestroyImmediate(renderTexture);
        
        Debug.Log("Tire image saved as: " + filename);
    }
    public IEnumerator CaptureTireImages()
    {

        for (int i = 0; i < tires.Count; i++)
        {
            yield return waitTimeBetweenTires;    
            currentTireIndex = i;
            tires[i].gameObject.SetActive(true);
            mainCamera.GetComponent<CameraOrbitalController>().center = tires[currentTireIndex].center;
    
            if (i > 0)
            {
                tires[i-1].gameObject.SetActive(false);
            }
            while (currentCount < count)
            {
                CaptureTireImage(count);
                yield return waitTime;
            }
            currentCount = 0;
            
        }
    }

}
