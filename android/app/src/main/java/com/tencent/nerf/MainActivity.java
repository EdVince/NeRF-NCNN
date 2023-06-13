// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.nerf;
import android.annotation.SuppressLint;
import android.content.Context;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Random;

public class MainActivity extends Activity
{
    private ImageView imageView;
    private Bitmap showBitmap;
    private String dataset = "Lego";


    private NeRF nerf = new NeRF();
    /** Called when the activity is first created. */
    @SuppressLint("MissingInflatedId")
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        imageView = (ImageView) findViewById(R.id.resView);
        showBitmap = Bitmap.createBitmap(800,800,Bitmap.Config.ARGB_8888);

        // copy assets data files
        String path = MainActivity.this.getFilesDir().getAbsolutePath();

        // copy bbox.txt intrinsics.txt model_density_bitfield.dat model_pos_encoder_hash_table.dat
        copy(MainActivity.this, dataset+File.separator+"bbox.txt", path, "bbox.txt");
        copy(MainActivity.this, dataset+File.separator+"intrinsics.txt", path, "intrinsics.txt");
        copy(MainActivity.this, dataset+File.separator+"model_density_bitfield.dat", path, "model_density_bitfield.dat");
        copy(MainActivity.this, dataset+File.separator+"model_pos_encoder_hash_table.dat", path, "model_pos_encoder_hash_table.dat");
        // copy 200 pose files
        for (int i = 0; i < 200; i++){
            String filename = "2_"+String.format("%04d",i)+".txt";
            copy(MainActivity.this, dataset+File.separator+"pose"+File.separator+filename, path, filename);
        }

        // Init NeRF
        nerf.Init(getAssets(), dataset, path);



        Button buttonTXT2IMG = (Button) findViewById(R.id.button);
        buttonTXT2IMG.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {

                        Random random = new Random();
                        int random_pose = random.nextInt(200);

                        nerf.render(showBitmap,random_pose);

                        final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                        imageView.post(new Runnable() {
                            public void run() {
                                imageView.setImageBitmap(styledImage);
                                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            }
                        });
                    }
                }).start();
            }
        });


    }

    private void copy(Context myContext, String ASSETS_NAME, String savePath, String saveName) {
        String filename = savePath + "/" + saveName;
        File dir = new File(savePath);
        if (!dir.exists())
            dir.mkdir();
        try {
            if (!(new File(filename)).exists()) {
                InputStream is = myContext.getResources().getAssets().open(ASSETS_NAME);
                FileOutputStream fos = new FileOutputStream(filename);
                byte[] buffer = new byte[7168];
                int count = 0;
                while ((count = is.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }
                fos.close();
                is.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
