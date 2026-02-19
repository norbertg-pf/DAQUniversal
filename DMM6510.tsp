-- Shared variable for sending data
local share_send_now = 0

function main()

	reset()

    -- Number of measurements
    local N = 10000
    -- Delay between readings (seconds)
    local dt = 0.0
	local v_treshold =  TRESHOLD --0.2e-3
	local QD = 0
    -- Create a reading buffer
    QDbuf = buffer.make(N, buffer.STYLE_WRITABLE_FULL)
	buffer.write.format(QDbuf, buffer.UNIT_VOLT, buffer.DIGITS_4_5)
    QDbuf.clear()
    QDbuf.fillmode = buffer.FILL_CONTINUOUS
	
	channel.open("allslots")	

    -- Configure the DMM to measure DC voltage
    dmm.measure.func = dmm.FUNC_DC_VOLTAGE
    dmm.measure.nplc = 0.1   -- integration period
	display.changescreen(display.SCREEN_GRAPH)
	display.activebuffer = QDbuf
    -- Take measurements
	timer.cleartime()
    -- Configure the DMM to measure DC voltage
-- Create scan buffer
	defbuffer1.clear()
	defbuffer1.capacity = 1000   -- adjust buffer size

	display.changescreen(display.SCREEN_GRAPH)
	display.activebuffer = QDbuf
    -- Take measurements
	
	trigger.extin.clear()
	trigger.extin.wait(1)
	
	timer.cleartime()


	-- Continuous loop
    while 1 do
        local t = timer.gettime()
		local v = dmm.measure.read()
        buffer.write.reading(QDbuf, v, t)  -- store timestamp + value
		-- Prepare data for LAN sender thread
    	share_data = string.format("%.9f", v)  -- time,value
    	print(share_data)
		delay(dt)                        -- wait
    end
end
-- Start the thread

main()
