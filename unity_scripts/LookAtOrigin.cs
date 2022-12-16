using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LookAtOrigin : MonoBehaviour
{

    public Vector3 pointToLook = new Vector3(0, 0, 0);

    // Start is called before the first frame update
    void Start()
    {
        transform.LookAt(pointToLook);
    }

    // Update is called once per frame
    void Update()
    {
        //do nothing
    }
}
