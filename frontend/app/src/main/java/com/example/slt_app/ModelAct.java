package com.example.slt_app;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class ModelAct extends AppCompatActivity {
    Button butn2;
    private static int VIDEO_REQUEST = 101;
    private Uri videoUri =null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.model_act);
        butn2 = findViewById(R.id.btn_model);
}
        public void traduire(View view)
        {
            Intent videoIntent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
            if (videoIntent.resolveActivity(getPackageManager() )!= null)
            {
                startActivityForResult(videoIntent,VIDEO_REQUEST);
            }
            Intent playIntent =new Intent (this,CamActivite.class);
            playIntent.putExtra("videoUri",videoUri.toString());
            startActivity(playIntent);
        }

        @Override
        protected void onActivityResult(int requestCode, int resultCode, Intent data) {

            super.onActivityResult(requestCode, resultCode, data);
            if ((requestCode == VIDEO_REQUEST && resultCode == RESULT_OK)) {
                videoUri = data.getData();
            }
        }
}
