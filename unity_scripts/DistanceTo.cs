using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DistanceTo : MonoBehaviour
{

    public Transform other;
    public float dist;

    // Start is called before the first frame update
    void Start()
    {
        dist = Vector3.Distance(other.position, transform.position);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
